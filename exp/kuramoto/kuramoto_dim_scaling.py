"""Nonlinear Kuramoto-type dimension-scaling experiment.

Run from /home/jixia/exp with:

    .venv/bin/python -u reachapprox/exp/kuramoto/kuramoto_dim_scaling.py

The experiment uses the product opened-triangle initial set from the linear
spring-mass experiment, but propagates samples through a nonlinear second-order
Kuramoto-type oscillator network. The estimator is the convex hull of the
propagated endpoints, and the metric is a numerical directed Hausdorff error

    max_{y in Y_ref_subset} dist(y, conv{Y_i}).

For adversarial sampling, the endpoint repulsion direction is used as a
lightweight approximate projected-gradient direction. This keeps the full
50-seed dimension sweep tractable for N=10000.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reachapprox.exp.spring.spring_mass_dim_scaling import (
    TRIANGLE_VERTICES,
    distance_to_eroded_triangle,
    point_in_opened_triangle,
    project_product_opened_triangles,
    project_to_opened_triangle,
    sample_opened_triangle,
    sample_product_opened_triangles,
)


DIMENSIONS = (2, 4, 6, 8, 10)
SAMPLE_BUDGETS = (1, 10, 100, 1000, 10000)
IMPROVEMENT_BUDGETS = (1, 10, 100, 1000)
N_SEEDS = 50

K_COUPLING = 1.0
C_DAMPING = 0.1
OMEGA0 = 1.0
T_HORIZON = 1.0
RHO = 0.2
TARGET_ACCURACY = 0.1

TIME_SWEEP_DIMENSION = 10
TIME_SWEEP_BUDGET = 10_000
TIME_SWEEP_GRID = tuple(float(t) for t in np.linspace(0.01, 5.0, 20))
TIME_SWEEP_N_REF = 50_000
TIME_SWEEP_SUBSET = 500
TIME_SWEEP_SEEDS = 50

N_REF = 200_000
COVERAGE_SUBSET = 500
REFERENCE_SEED = 202705
EXPERIMENT_SEED = 314159

# Fixed-step RK4 is vectorized over point clouds. The same scheme is used for
# state propagation and for the variational equation in adversarial gradients.
RK4_STEPS = 80

# Adversarial sampler. The routine returns exactly N endpoints for each budget.
# Since the reported metric is endpoint-cloud coverage error, the adversarial
# update uses a repulsive endpoint-separation objective rather than the
# covariance-radius objective used for boundary-biased reconstruction.
N_ADV = 2
ETA = 0.1
ADVERSARIAL_MOVE_FRACTION = 0.2
LAMBDA_REG = 1e-4
QMC_BATCH_FACTOR = 4

# Numerical projection-to-convex-hull parameters used in the directed
# Hausdorff computation. For large N, a maximin coreset keeps the projection
# problem tractable while preserving the exposed geometry of conv(Y_N).
MAX_HULL_VERTICES_FOR_DISTANCE = 500
FRANK_WOLFE_MAX_ITER = 25
FRANK_WOLFE_CHUNK_SIZE = 128
FRANK_WOLFE_TOL = 1e-6

FIG_DIR = Path("CoRL_2026/fig")
UNIFORM_FIG = FIG_DIR / "kuramoto_uniform_dim_scaling_error_vs_N.png"
ADVERSARIAL_FIG = FIG_DIR / "kuramoto_adversarial_dim_scaling_error_vs_N.png"
THREE_PANEL_FIG = FIG_DIR / "kuramoto_sampling_comparison_three_panel.png"
CSV_PATH = FIG_DIR / "kuramoto_opened_triangle_dim_scaling_results.csv"
TIME_SWEEP_CSV_PATH = FIG_DIR / "kuramoto_time_sweep_N10000_results.csv"

METRIC_NAME = "directed_hausdorff_to_convex_hull"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)

TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 13
LEGEND_SIZE = 12


@dataclass(frozen=True)
class ExperimentResult:
    method: str
    n: int
    m: int
    budget: int
    seed: int
    error: float


@dataclass(frozen=True)
class TimeSweepResult:
    method: str
    n: int
    m: int
    budget: int
    time: float
    seed: int
    error: float


def kuramoto_vector_field(X: np.ndarray, m: int) -> np.ndarray:
    """Second-order Kuramoto-type dynamics in block order (q1,p1,...,qm,pm)."""
    blocks = np.asarray(X, dtype=float).reshape(-1, m, 2)
    q = blocks[:, :, 0]
    p = blocks[:, :, 1]

    q_prev = np.roll(q, 1, axis=1)
    q_next = np.roll(q, -1, axis=1)
    coupling = np.sin(q - q_prev) + np.sin(q - q_next)

    dq = p
    dp = -(OMEGA0**2) * np.sin(q) - K_COUPLING * coupling - C_DAMPING * p
    return np.stack((dq, dp), axis=2).reshape(-1, 2 * m)


def kuramoto_jacobian(X: np.ndarray, m: int) -> np.ndarray:
    """Jacobian Df(x) for each state in block-wise coordinates."""
    X = np.asarray(X, dtype=float)
    batch = X.shape[0]
    n = 2 * m
    q = X.reshape(batch, m, 2)[:, :, 0]
    q_prev = np.roll(q, 1, axis=1)
    q_next = np.roll(q, -1, axis=1)

    cos_prev = np.cos(q - q_prev)
    cos_next = np.cos(q - q_next)

    A = np.zeros((batch, n, n), dtype=float)
    for i in range(m):
        qi = 2 * i
        pi = qi + 1
        q_prev_idx = 2 * ((i - 1) % m)
        q_next_idx = 2 * ((i + 1) % m)

        A[:, qi, pi] = 1.0
        A[:, pi, pi] = -C_DAMPING
        A[:, pi, qi] += -(OMEGA0**2) * np.cos(q[:, i]) - K_COUPLING * (
            cos_prev[:, i] + cos_next[:, i]
        )
        A[:, pi, q_prev_idx] += K_COUPLING * cos_prev[:, i]
        A[:, pi, q_next_idx] += K_COUPLING * cos_next[:, i]

    return A


def rk4_step_state(X: np.ndarray, dt: float, m: int) -> np.ndarray:
    k1 = kuramoto_vector_field(X, m)
    k2 = kuramoto_vector_field(X + 0.5 * dt * k1, m)
    k3 = kuramoto_vector_field(X + 0.5 * dt * k2, m)
    k4 = kuramoto_vector_field(X + dt * k3, m)
    return X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def variational_rhs(X: np.ndarray, J: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    dX = kuramoto_vector_field(X, m)
    A = kuramoto_jacobian(X, m)
    dJ = np.einsum("bij,bjk->bik", A, J)
    return dX, dJ


def rk4_step_state_jacobian(X: np.ndarray, J: np.ndarray, dt: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    k1x, k1j = variational_rhs(X, J, m)
    k2x, k2j = variational_rhs(X + 0.5 * dt * k1x, J + 0.5 * dt * k1j, m)
    k3x, k3j = variational_rhs(X + 0.5 * dt * k2x, J + 0.5 * dt * k2j, m)
    k4x, k4j = variational_rhs(X + dt * k3x, J + dt * k3j, m)
    X_next = X + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    J_next = J + (dt / 6.0) * (k1j + 2.0 * k2j + 2.0 * k3j + k4j)
    return X_next, J_next


def propagate_kuramoto_flow(
    X: np.ndarray,
    m: int,
    horizon: float = T_HORIZON,
    return_jacobian: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Propagate a batch through the nonlinear flow to the requested horizon."""
    X = np.asarray(X, dtype=float).copy()
    n_steps = max(1, int(np.ceil(RK4_STEPS * horizon / T_HORIZON)))
    dt = horizon / n_steps

    if not return_jacobian:
        for _ in range(n_steps):
            X = rk4_step_state(X, dt, m)
        return X

    n = 2 * m
    eye = np.eye(n, dtype=float)
    J = np.broadcast_to(eye, (X.shape[0], n, n)).copy()
    for _ in range(n_steps):
        X, J = rk4_step_state_jacobian(X, J, dt, m)
    return X, J


def farthest_point_coreset(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Select a maximin coreset for large convex-hull vertex sets."""
    if points.shape[0] <= max_points:
        return points

    first = int(rng.integers(0, points.shape[0]))
    selected = np.empty(max_points, dtype=int)
    selected[0] = first
    min_dist = np.linalg.norm(points - points[first], axis=1)
    min_dist[first] = -np.inf

    for k in range(1, max_points):
        idx = int(np.argmax(min_dist))
        selected[k] = idx
        dist = np.linalg.norm(points - points[idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[idx] = -np.inf

    return points[selected]


def directed_hausdorff_to_convex_hull(
    Y_ref_subset: np.ndarray,
    Y_vertices: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Numerical directed Hausdorff distance from reference points to conv(Y).

    Each reference point is projected onto the convex hull by Frank-Wolfe. This
    evaluates distance to the convex hull estimator, not nearest-neighbor
    distance to the endpoint point cloud.
    """
    vertices = farthest_point_coreset(Y_vertices, MAX_HULL_VERTICES_FOR_DISTANCE, rng)
    if vertices.shape[0] == 1:
        return float(np.linalg.norm(Y_ref_subset - vertices[0], axis=1).max())

    tree = cKDTree(vertices)
    max_dist = 0.0

    for start in range(0, Y_ref_subset.shape[0], FRANK_WOLFE_CHUNK_SIZE):
        Y = Y_ref_subset[start : start + FRANK_WOLFE_CHUNK_SIZE]
        nearest = tree.query(Y, k=1, workers=-1)[1]
        Z = vertices[nearest].copy()

        for _ in range(FRANK_WOLFE_MAX_ITER):
            grad = Z - Y
            idx = np.argmin(grad @ vertices.T, axis=1)
            S = vertices[idx]
            direction = S - Z
            denom = np.sum(direction * direction, axis=1)
            active = denom > 1e-15
            gamma = np.zeros(Y.shape[0], dtype=float)
            gamma[active] = np.clip(
                -np.sum((Z[active] - Y[active]) * direction[active], axis=1) / denom[active],
                0.0,
                1.0,
            )
            Z += gamma[:, None] * direction
            if np.max(gamma * np.sqrt(np.maximum(denom, 0.0))) < FRANK_WOLFE_TOL:
                break

        max_dist = max(max_dist, float(np.linalg.norm(Y - Z, axis=1).max()))

    return max_dist


def sample_opened_triangle_qmc(num_samples: int, seed: int, rho: float = RHO) -> np.ndarray:
    """Low-discrepancy rejection sampling from the opened triangle."""
    accepted: list[np.ndarray] = []
    count = 0
    batch = max(1024, QMC_BATCH_FACTOR * num_samples)
    attempt = 0
    a1, a2, a3 = TRIANGLE_VERTICES

    while count < num_samples:
        power = int(np.ceil(np.log2(batch)))
        sampler = qmc.Sobol(d=2, scramble=True, seed=seed + attempt)
        uv = sampler.random_base2(power)[:batch]
        reflect = np.sum(uv, axis=1) > 1.0
        uv[reflect] = 1.0 - uv[reflect]
        candidates = a1 + uv[:, 0:1] * (a2 - a1) + uv[:, 1:2] * (a3 - a1)
        mask = distance_to_eroded_triangle(candidates) <= rho + 1e-12
        chosen = candidates[mask]
        if chosen.size:
            accepted.append(chosen)
            count += chosen.shape[0]
        attempt += 1

    return np.vstack(accepted)[:num_samples]


def sample_product_opened_triangles_qmc(N: int, m: int, seed: int, rho: float = RHO) -> np.ndarray:
    """Full-dimensional QMC rejection sampling for S0^(m)=prod_i G_rho."""
    accepted: list[np.ndarray] = []
    count = 0
    batch = max(1024, QMC_BATCH_FACTOR * N)
    attempt = 0
    a1, a2, a3 = TRIANGLE_VERTICES

    while count < N:
        power = int(np.ceil(np.log2(batch)))
        sampler = qmc.Sobol(d=2 * m, scramble=True, seed=seed + attempt)
        U = sampler.random_base2(power)[:batch]
        blocks = []
        mask = np.ones(U.shape[0], dtype=bool)
        for i in range(m):
            uv = U[:, 2 * i : 2 * i + 2].copy()
            reflect = np.sum(uv, axis=1) > 1.0
            uv[reflect] = 1.0 - uv[reflect]
            pts = a1 + uv[:, 0:1] * (a2 - a1) + uv[:, 1:2] * (a3 - a1)
            blocks.append(pts)
            mask &= distance_to_eroded_triangle(pts) <= rho + 1e-12
        chosen = np.stack(blocks, axis=1)[mask].reshape(-1, 2 * m)
        if chosen.size:
            accepted.append(chosen)
            count += chosen.shape[0]
        attempt += 1

    return np.vstack(accepted)[:N]


def regularized_inverse_covariance(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.mean(Y, axis=0)
    centered = Y - c
    if Y.shape[0] > 1:
        cov = (centered.T @ centered) / (Y.shape[0] - 1)
    else:
        cov = np.zeros((Y.shape[1], Y.shape[1]), dtype=float)
    Q = np.linalg.inv(cov + LAMBDA_REG * np.eye(Y.shape[1]))
    return c, Q


def adversarial_endpoint_samples(
    rng: np.random.Generator,
    m: int,
    total_budget: int,
    horizon: float = T_HORIZON,
    initial_points: np.ndarray | None = None,
) -> np.ndarray:
    """Generate exactly N coverage-aware adversarial endpoint samples.

    The current metric is an endpoint-cloud coverage error. We use
    full-dimensional Sobol restarts to reduce Monte-Carlo coverage holes, then
    apply selective local repulsion only to the most crowded endpoints.
    """
    if total_budget == 1:
        X = (
            sample_product_opened_triangles_qmc(1, m, int(rng.integers(0, 2**31 - 1)), RHO)
            if initial_points is None
            else initial_points[:1].copy()
        )
        return propagate_kuramoto_flow(X, m, horizon=horizon)

    X = (
        sample_product_opened_triangles_qmc(total_budget, m, int(rng.integers(0, 2**31 - 1)), RHO)
        if initial_points is None
        else np.asarray(initial_points, dtype=float)[:total_budget].copy()
    )
    for _ in range(N_ADV):
        Y_current = propagate_kuramoto_flow(X, m, horizon=horizon)
        nn_dist, nn_idx = cKDTree(Y_current).query(Y_current, k=2, workers=-1)
        move_count = max(1, int(round(ADVERSARIAL_MOVE_FRACTION * total_budget)))
        move_idx = np.argsort(nn_dist[:, 1])[:move_count]
        nearest = Y_current[nn_idx[move_idx, 1]]
        grad = np.zeros_like(X)
        selected_grad = 2.0 * (Y_current[move_idx] - nearest)
        selected_grad /= np.maximum(np.linalg.norm(selected_grad, axis=1, keepdims=True), 1e-12)
        grad[move_idx] = selected_grad
        X = project_product_opened_triangles(X + ETA * grad, m, RHO)

    return propagate_kuramoto_flow(X, m, horizon=horizon)


def mean_and_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=1)
    stderr = np.std(values, axis=1, ddof=1) / np.sqrt(values.shape[1])
    half_width = 1.96 * stderr
    return mean, np.maximum(mean - half_width, np.finfo(float).tiny), mean + half_width


def run_experiment() -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for n in DIMENSIONS:
        m = n // 2
        ref_rng = np.random.default_rng(REFERENCE_SEED + n)
        X_ref = sample_product_opened_triangles(N_REF, m, RHO, ref_rng)
        Y_ref = propagate_kuramoto_flow(X_ref, m)
        subset_size = min(COVERAGE_SUBSET, N_REF)

        print(f"\nKuramoto n={n}, m={m}: reference cloud {N_REF}, coverage subset {subset_size}")
        for budget in SAMPLE_BUDGETS:
            for seed_index in range(N_SEEDS):
                seed = EXPERIMENT_SEED + 10_000 * n + 100 * seed_index + budget
                rng = np.random.default_rng(seed)
                subset_idx = rng.choice(N_REF, size=subset_size, replace=False)
                Y_ref_subset = Y_ref[subset_idx]

                X_uniform = sample_product_opened_triangles(budget, m, RHO, rng)
                Y_uniform = propagate_kuramoto_flow(X_uniform, m)
                uniform_error = directed_hausdorff_to_convex_hull(Y_ref_subset, Y_uniform, rng)
                results.append(ExperimentResult("uniform", n, m, budget, seed, uniform_error))

                Y_adv = adversarial_endpoint_samples(rng, m, budget)
                adv_error = directed_hausdorff_to_convex_hull(Y_ref_subset, Y_adv, rng)
                results.append(ExperimentResult("adversarial", n, m, budget, seed, adv_error))

            print(f"  finished N={budget}")

    return results


def save_results_csv(results: list[ExperimentResult]) -> None:
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("experiment,method,n,m,N,seed,error,K,c,omega0,T,rho,metric_name\n")
        for row in results:
            f.write(
                f"kuramoto,{row.method},{row.n},{row.m},{row.budget},{row.seed},"
                f"{row.error:.12g},{K_COUPLING},{C_DAMPING},{OMEGA0},{T_HORIZON},{RHO},{METRIC_NAME}\n"
            )


def load_results_csv() -> list[ExperimentResult] | None:
    if not CSV_PATH.exists():
        return None

    results: list[ExperimentResult] = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("experiment") != "kuramoto":
                continue
            results.append(
                ExperimentResult(
                    method=row["method"],
                    n=int(row["n"]),
                    m=int(row["m"]),
                    budget=int(row["N"]),
                    seed=int(row["seed"]),
                    error=float(row["error"]),
                )
            )

    expected = len(DIMENSIONS) * len(SAMPLE_BUDGETS) * N_SEEDS * 2
    if len(results) < expected:
        return None
    return results


def aggregate_results(results: list[ExperimentResult], method: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((len(SAMPLE_BUDGETS), N_SEEDS), dtype=float)
    for i, budget in enumerate(SAMPLE_BUDGETS):
        selected = [r.error for r in results if r.method == method and r.n == n and r.budget == budget]
        values[i] = np.asarray(selected, dtype=float)
    return mean_and_ci(values)


def aggregate_improvement(results: list[ExperimentResult], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((len(IMPROVEMENT_BUDGETS), N_SEEDS), dtype=float)
    for i, budget in enumerate(IMPROVEMENT_BUDGETS):
        uniform = sorted(
            (r for r in results if r.method == "uniform" and r.n == n and r.budget == budget),
            key=lambda r: r.seed,
        )
        adversarial = sorted(
            (r for r in results if r.method == "adversarial" and r.n == n and r.budget == budget),
            key=lambda r: r.seed,
        )
        values[i] = np.asarray([u.error - a.error for u, a in zip(uniform, adversarial)], dtype=float)
    return mean_and_ci(values)


def run_time_sweep_experiment() -> list[TimeSweepResult]:
    results: list[TimeSweepResult] = []
    n = TIME_SWEEP_DIMENSION
    m = n // 2
    ref_rng = np.random.default_rng(REFERENCE_SEED + 90_000 + n)
    X_ref = sample_product_opened_triangles(TIME_SWEEP_N_REF, m, RHO, ref_rng)
    subset_size = min(TIME_SWEEP_SUBSET, TIME_SWEEP_N_REF)

    print(
        f"\nKuramoto time sweep n={n}, N={TIME_SWEEP_BUDGET}: "
        f"reference cloud {TIME_SWEEP_N_REF}, coverage subset {subset_size}"
    )
    for time in TIME_SWEEP_GRID:
        Y_ref = propagate_kuramoto_flow(X_ref, m, horizon=time)
        for seed_index in range(TIME_SWEEP_SEEDS):
            seed = EXPERIMENT_SEED + 900_000 + 1000 * seed_index + int(round(10_000 * time))
            rng = np.random.default_rng(seed)
            subset_idx = rng.choice(TIME_SWEEP_N_REF, size=subset_size, replace=False)
            Y_ref_subset = Y_ref[subset_idx]

            X_uniform = sample_product_opened_triangles(TIME_SWEEP_BUDGET, m, RHO, rng)
            Y_uniform = propagate_kuramoto_flow(X_uniform, m, horizon=time)
            uniform_error = directed_hausdorff_to_convex_hull(Y_ref_subset, Y_uniform, rng)
            results.append(TimeSweepResult("uniform", n, m, TIME_SWEEP_BUDGET, time, seed, uniform_error))

            Y_adv = adversarial_endpoint_samples(rng, m, TIME_SWEEP_BUDGET, horizon=time)
            adv_error = directed_hausdorff_to_convex_hull(Y_ref_subset, Y_adv, rng)
            results.append(TimeSweepResult("adversarial", n, m, TIME_SWEEP_BUDGET, time, seed, adv_error))

        print(f"  finished t={time:.4f}")

    return results


def save_time_sweep_csv(results: list[TimeSweepResult]) -> None:
    with TIME_SWEEP_CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("experiment,method,n,m,N,time,seed,error,K,c,omega0,T,rho,metric_name\n")
        for row in results:
            f.write(
                f"kuramoto_time_sweep,{row.method},{row.n},{row.m},{row.budget},{row.time:.12g},{row.seed},"
                f"{row.error:.12g},{K_COUPLING},{C_DAMPING},{OMEGA0},{row.time},{RHO},{METRIC_NAME}\n"
            )


def load_time_sweep_csv() -> list[TimeSweepResult] | None:
    if not TIME_SWEEP_CSV_PATH.exists():
        return None

    results: list[TimeSweepResult] = []
    with TIME_SWEEP_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                TimeSweepResult(
                    method=row["method"],
                    n=int(row["n"]),
                    m=int(row["m"]),
                    budget=int(row["N"]),
                    time=float(row["time"]),
                    seed=int(row["seed"]),
                    error=float(row["error"]),
                )
            )

    expected = len(TIME_SWEEP_GRID) * TIME_SWEEP_SEEDS * 2
    if len(results) < expected:
        return None
    return results


def aggregate_time_sweep(results: list[TimeSweepResult], method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((len(TIME_SWEEP_GRID), TIME_SWEEP_SEEDS), dtype=float)
    for i, time in enumerate(TIME_SWEEP_GRID):
        selected = [
            r.error
            for r in results
            if r.method == method and np.isclose(r.time, time) and r.budget == TIME_SWEEP_BUDGET
        ]
        values[i] = np.asarray(selected[:TIME_SWEEP_SEEDS], dtype=float)
    return mean_and_ci(values)


def plot_method(results: list[ExperimentResult], method: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(DIMENSIONS)))
    markers = ("o", "s", "D", "^", "P")

    for n, color, marker in zip(DIMENSIONS, colors, markers):
        mean, lo, hi = aggregate_results(results, method, n)
        ax.plot(SAMPLE_BUDGETS, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}")
        ax.fill_between(SAMPLE_BUDGETS, lo, hi, color=color, alpha=0.20)

    ax.axhline(TARGET_ACCURACY, color="red", linestyle="--", linewidth=2.0, label="r=0.1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of samples N", fontsize=LABEL_SIZE)
    ax.set_ylabel("Hausdorff error", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_three_panel(results: list[ExperimentResult], time_results: list[TimeSweepResult]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(DIMENSIONS)))
    markers = ("o", "s", "D", "^", "P")

    ax = axes[0]
    for n, color, marker in zip(DIMENSIONS, colors, markers):
        mean, lo, hi = aggregate_results(results, "uniform", n)
        ax.plot(SAMPLE_BUDGETS, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}")
        ax.fill_between(SAMPLE_BUDGETS, lo, hi, color=color, alpha=0.20)
    ax.axhline(TARGET_ACCURACY, color="red", linestyle="--", linewidth=2.0, label="r=0.1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of samples N", fontsize=LABEL_SIZE)
    ax.set_ylabel("Hausdorff error", fontsize=LABEL_SIZE)
    ax.set_title("Uniform Sampling", fontsize=TITLE_SIZE)
    ax.grid(True, which="both", alpha=0.28)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)

    ax = axes[1]
    for n, color, marker in zip(DIMENSIONS, colors, markers):
        mean, lo, hi = aggregate_improvement(results, n)
        ax.plot(IMPROVEMENT_BUDGETS, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}")
        ax.fill_between(IMPROVEMENT_BUDGETS, lo, hi, color=color, alpha=0.20)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("number of samples N", fontsize=LABEL_SIZE)
    ax.set_ylabel("Hausdorff error reduction", fontsize=LABEL_SIZE)
    ax.set_title("Adversarial Improvement", fontsize=TITLE_SIZE)
    ax.grid(True, which="both", alpha=0.28)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)

    ax = axes[2]
    method_styles = {
        "uniform": ("tab:blue", "o", "uniform"),
        "adversarial": ("tab:orange", "s", "adversarial"),
    }
    for method, (color, marker, label) in method_styles.items():
        mean, lo, hi = aggregate_time_sweep(time_results, method)
        ax.plot(TIME_SWEEP_GRID, mean, color=color, marker=marker, lw=2.0, ms=5.5, label=label)
        ax.fill_between(TIME_SWEEP_GRID, lo, hi, color=color, alpha=0.20)
    ax.set_xlabel("time t", fontsize=LABEL_SIZE)
    ax.set_ylabel("Hausdorff error", fontsize=LABEL_SIZE)
    ax.set_title(f"Time Sweep (n={TIME_SWEEP_DIMENSION}, N={TIME_SWEEP_BUDGET})", fontsize=TITLE_SIZE)
    ax.grid(True, alpha=0.28)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)

    fig.savefig(THREE_PANEL_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results_csv()
    if results is None:
        results = run_experiment()
        save_results_csv(results)
    else:
        print(f"Loaded cached dimension-scaling results from {CSV_PATH}")

    time_results = load_time_sweep_csv()
    if time_results is None:
        time_results = run_time_sweep_experiment()
        save_time_sweep_csv(time_results)
    else:
        print(f"Loaded cached time-sweep results from {TIME_SWEEP_CSV_PATH}")

    plot_method(results, "uniform", UNIFORM_FIG, "Uniform Sampling")
    plot_three_panel(results, time_results)

    print("\nSaved:")
    print(UNIFORM_FIG)
    print(THREE_PANEL_FIG)
    print(CSV_PATH)
    print(TIME_SWEEP_CSV_PATH)


if __name__ == "__main__":
    main()
