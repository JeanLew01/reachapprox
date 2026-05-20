"""Spring-mass-damper dimension-scaling experiment.

Run from /home/jixia/exp with:

    .venv/bin/python -u reachapprox/exp/spring/spring_mass_dim_scaling.py

The reachable-set estimator is the convex hull of the endpoint samples. Since
exact point-to-convex-hull distances are expensive in dimensions up to n=10 and
N=10000, the metric uses a sampled convex-hull reconstruction cloud:

    max_{y in Y_ref_subset} min_{z in sampled conv(Y_i)} ||y - z||.

This is an empirical directed Hausdorff / coverage error for the convex-hull
estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.linalg import expm
from scipy.spatial import cKDTree


DIMENSIONS = (2, 4, 6, 8, 10)
SAMPLE_BUDGETS = (1, 10, 100, 1000, 10000)
N_SEEDS = 50

K_SPRING = 1.0
C_DAMPING = 0.2
T_HORIZON = 1.0
RHO = 0.2
TARGET_ACCURACY = 0.01

N_REF = 200_000
COVERAGE_SUBSET = 20_000
HULL_RECONSTRUCTION_SIZE = 50_000
REFERENCE_SEED = 202604
EXPERIMENT_SEED = 90917

# Adversarial sampler. For each budget, the code returns exactly N endpoints.
# For the convex-hull estimator, a small number of adversarial updates keeps the
# samples diverse while still pushing endpoints toward exposed directions.
# For N=1, the adversarial sampler returns one random point.
N_ADV = 2
ETA = 0.2
LAMBDA_REG = 1e-4

FIG_DIR = Path("CoRL_2026/fig")
UNIFORM_FIG = FIG_DIR / "spring_mass_uniform_dim_scaling_error_vs_N.png"
ADVERSARIAL_FIG = FIG_DIR / "spring_mass_adversarial_dim_scaling_error_vs_N.png"
CSV_PATH = FIG_DIR / "spring_mass_opened_triangle_dim_scaling_results.csv"

METRIC_NAME = "empirical_directed_hausdorff_sampled_convex_hull"

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


SQRT3 = np.sqrt(3.0)
TRIANGLE_VERTICES = np.array(
    [
        [1.0, 0.0],
        [-0.5, SQRT3 / 2.0],
        [-0.5, -SQRT3 / 2.0],
    ],
    dtype=float,
)
ERODED_VERTICES = 0.6 * TRIANGLE_VERTICES
TRIANGLE_PATH = MplPath(TRIANGLE_VERTICES)
ERODED_PATH = MplPath(ERODED_VERTICES)


@dataclass(frozen=True)
class ExperimentResult:
    method: str
    n: int
    m: int
    budget: int
    seed: int
    error: float


def point_to_segments_distance(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Distance from points to the boundary segments of a convex polygon."""
    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    segs = ends - starts
    seg_norm2 = np.sum(segs * segs, axis=1)
    diff = points[:, None, :] - starts[None, :, :]
    tau = np.einsum("nsi,si->ns", diff, segs) / seg_norm2[None, :]
    tau = np.clip(tau, 0.0, 1.0)
    closest = starts[None, :, :] + tau[:, :, None] * segs[None, :, :]
    dist2 = np.sum((points[:, None, :] - closest) ** 2, axis=2)
    return np.sqrt(np.min(dist2, axis=1))


def distance_to_eroded_triangle(points: np.ndarray) -> np.ndarray:
    """Distance to the eroded triangle 0.6*T."""
    inside = ERODED_PATH.contains_points(points, radius=1e-12)
    distances = point_to_segments_distance(points, ERODED_VERTICES)
    distances[inside] = 0.0
    return distances


def point_in_opened_triangle(z: np.ndarray, rho: float = RHO) -> bool:
    """Membership in G_rho = (0.6*T) + B_rho."""
    point = np.asarray(z, dtype=float).reshape(1, 2)
    return bool(distance_to_eroded_triangle(point)[0] <= rho + 1e-12)


def project_to_convex_polygon(points: np.ndarray, vertices: np.ndarray, path: MplPath) -> np.ndarray:
    """Project points to a convex polygon using nearest boundary points for outsiders."""
    inside = path.contains_points(points, radius=1e-12)
    projected = points.copy()
    if np.all(inside):
        return projected

    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    segs = ends - starts
    seg_norm2 = np.sum(segs * segs, axis=1)
    outside_points = points[~inside]
    diff = outside_points[:, None, :] - starts[None, :, :]
    tau = np.einsum("nsi,si->ns", diff, segs) / seg_norm2[None, :]
    tau = np.clip(tau, 0.0, 1.0)
    candidates = starts[None, :, :] + tau[:, :, None] * segs[None, :, :]
    dist2 = np.sum((outside_points[:, None, :] - candidates) ** 2, axis=2)
    nearest = np.argmin(dist2, axis=1)
    projected[~inside] = candidates[np.arange(outside_points.shape[0]), nearest]
    return projected


def project_to_opened_triangle(z: np.ndarray, rho: float = RHO) -> np.ndarray:
    """Project a point to G_rho = (0.6*T) + B_rho."""
    point = np.asarray(z, dtype=float).reshape(1, 2)
    projected = project_opened_triangle_batch(point, rho)
    return projected[0]


def project_opened_triangle_batch(points: np.ndarray, rho: float = RHO) -> np.ndarray:
    """Vectorized projection to the rho-offset of the eroded triangle."""
    p = project_to_convex_polygon(points, ERODED_VERTICES, ERODED_PATH)
    diff = points - p
    dist = np.linalg.norm(diff, axis=1)
    out = points.copy()
    outside = dist > rho
    out[outside] = p[outside] + rho * diff[outside] / dist[outside, None]
    return out


def sample_opened_triangle(num_samples: int, rho: float = RHO, rng: np.random.Generator | None = None) -> np.ndarray:
    """Approximately uniform rejection sampling from G_rho.

    We sample uniformly from the original triangle T, then accept points whose
    distance to the eroded triangle 0.6*T is at most rho.
    """
    rng = np.random.default_rng() if rng is None else rng
    accepted: list[np.ndarray] = []
    count = 0
    batch = max(1024, 2 * num_samples)

    while count < num_samples:
        weights = rng.exponential(1.0, size=(batch, 3))
        weights /= np.sum(weights, axis=1, keepdims=True)
        candidates = weights @ TRIANGLE_VERTICES
        mask = distance_to_eroded_triangle(candidates) <= rho + 1e-12
        chosen = candidates[mask]
        if chosen.size:
            accepted.append(chosen)
            count += chosen.shape[0]

    return np.vstack(accepted)[:num_samples]


def sample_product_opened_triangles(
    N: int,
    m: int,
    rho: float = RHO,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample S0^(m) = product_i G_rho in block-wise order (q1,v1,...,qm,vm)."""
    rng = np.random.default_rng() if rng is None else rng
    blocks = sample_opened_triangle(N * m, rho, rng).reshape(N, m, 2)
    return blocks.reshape(N, 2 * m)


def project_product_opened_triangles(X: np.ndarray, m: int, rho: float = RHO) -> np.ndarray:
    """Project block-wise onto the product initial set."""
    blocks = np.asarray(X, dtype=float).reshape(-1, m, 2)
    projected = project_opened_triangle_batch(blocks.reshape(-1, 2), rho)
    return projected.reshape(-1, 2 * m)


def block_to_qv_order(X: np.ndarray, m: int) -> np.ndarray:
    """Convert (q1,v1,...,qm,vm) to (q1,...,qm,v1,...,vm)."""
    blocks = np.asarray(X, dtype=float).reshape(-1, m, 2)
    return np.hstack((blocks[:, :, 0], blocks[:, :, 1]))


def qv_to_block_order(X: np.ndarray, m: int) -> np.ndarray:
    """Convert (q1,...,qm,v1,...,vm) to (q1,v1,...,qm,vm)."""
    X = np.asarray(X, dtype=float)
    q = X[:, :m]
    v = X[:, m:]
    return np.stack((q, v), axis=2).reshape(-1, 2 * m)


def build_spring_mass_matrix(m: int, k: float = K_SPRING, c: float = C_DAMPING) -> np.ndarray:
    """Build A in block-wise state ordering x=(q1,v1,...,qm,vm)."""
    lap = np.zeros((m, m), dtype=float)
    for i in range(m):
        lap[i, i] = 2.0
        lap[i, (i - 1) % m] = -1.0
        lap[i, (i + 1) % m] = -1.0

    # Internal matrix in (q1,...,qm,v1,...,vm) order.
    A_qv = np.block(
        [
            [np.zeros((m, m)), np.eye(m)],
            [-k * lap, -c * np.eye(m)],
        ]
    )

    # Permute to block-wise order.
    n = 2 * m
    perm = []
    for i in range(m):
        perm.extend([i, m + i])
    P = np.eye(n)[perm]
    return P @ A_qv @ P.T


def propagate_linear_flow(X: np.ndarray, Phi_T: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float) @ Phi_T.T


def empirical_directed_hausdorff(Y_ref_subset: np.ndarray, Y_samples: np.ndarray) -> float:
    tree = cKDTree(Y_samples)
    return float(tree.query(Y_ref_subset, k=1, workers=-1)[0].max())


def sample_convex_hull_cloud(
    points: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a point cloud from conv(points) using random convex combinations.

    By Caratheodory's theorem, points in R^n can be represented using at most
    n+1 vertices. We use random subsets of that size and Dirichlet weights to
    build a tractable convex-hull reconstruction cloud for the coverage metric.
    """
    points = np.asarray(points, dtype=float)
    n_samples, dim = points.shape
    if n_samples == 1:
        return points.copy()

    combo_size = min(dim + 1, n_samples)
    idx = rng.integers(0, n_samples, size=(num_points, combo_size))
    weights = rng.exponential(1.0, size=(num_points, combo_size))
    weights /= np.sum(weights, axis=1, keepdims=True)
    hull_cloud = np.einsum("kc,kcd->kd", weights, points[idx])
    return np.vstack((points, hull_cloud))


def empirical_directed_hausdorff_to_convex_hull(
    Y_ref_subset: np.ndarray,
    Y_samples: np.ndarray,
    rng: np.random.Generator,
) -> float:
    hull_cloud = sample_convex_hull_cloud(Y_samples, HULL_RECONSTRUCTION_SIZE, rng)
    return empirical_directed_hausdorff(Y_ref_subset, hull_cloud)


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
    Phi_T: np.ndarray,
    total_budget: int,
) -> np.ndarray:
    """Generate exactly N adversarial endpoint samples.

    Use M=ceil(N/(N_ADV+1)) particles and truncate the accumulated cloud to
    exactly N endpoints. For N=1, return one random endpoint.
    """
    if total_budget == 1:
        X = sample_product_opened_triangles(1, m, RHO, rng)
        return propagate_linear_flow(X, Phi_T)

    M = int(np.ceil(total_budget / (N_ADV + 1)))
    X = sample_product_opened_triangles(M, m, RHO, rng)
    endpoints = [propagate_linear_flow(X, Phi_T)]

    for _ in range(N_ADV):
        Y_acc = np.vstack(endpoints)
        center, Q = regularized_inverse_covariance(Y_acc)
        Y_current = propagate_linear_flow(X, Phi_T)
        grad = 2.0 * (Y_current - center) @ Q.T @ Phi_T
        X = project_product_opened_triangles(X + ETA * grad, m, RHO)
        endpoints.append(propagate_linear_flow(X, Phi_T))

    return np.vstack(endpoints)[:total_budget]


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
        A = build_spring_mass_matrix(m, K_SPRING, C_DAMPING)
        Phi_T = expm(A * T_HORIZON)
        ref_rng = np.random.default_rng(REFERENCE_SEED + n)
        X_ref = sample_product_opened_triangles(N_REF, m, RHO, ref_rng)
        Y_ref = propagate_linear_flow(X_ref, Phi_T)
        subset_size = min(COVERAGE_SUBSET, N_REF)

        print(f"\nDimension n={n}, m={m}: reference cloud {N_REF}, coverage subset {subset_size}")
        for budget in SAMPLE_BUDGETS:
            for seed_index in range(N_SEEDS):
                seed = EXPERIMENT_SEED + 10_000 * n + 100 * seed_index + budget
                rng = np.random.default_rng(seed)
                subset_idx = rng.choice(N_REF, size=subset_size, replace=False)
                Y_ref_subset = Y_ref[subset_idx]

                X_uniform = sample_product_opened_triangles(budget, m, RHO, rng)
                Y_uniform = propagate_linear_flow(X_uniform, Phi_T)
                uniform_error = empirical_directed_hausdorff_to_convex_hull(Y_ref_subset, Y_uniform, rng)
                results.append(ExperimentResult("uniform", n, m, budget, seed, uniform_error))

                Y_adv = adversarial_endpoint_samples(rng, m, Phi_T, budget)
                adv_error = empirical_directed_hausdorff_to_convex_hull(Y_ref_subset, Y_adv, rng)
                results.append(ExperimentResult("adversarial", n, m, budget, seed, adv_error))

            print(f"  finished N={budget}")

    return results


def save_results_csv(results: list[ExperimentResult]) -> None:
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("method,n,m,N,seed,error,k,c,T,rho,metric_name\n")
        for row in results:
            f.write(
                f"{row.method},{row.n},{row.m},{row.budget},{row.seed},"
                f"{row.error:.12g},{K_SPRING},{C_DAMPING},{T_HORIZON},{RHO},{METRIC_NAME}\n"
            )


def aggregate_results(results: list[ExperimentResult], method: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((len(SAMPLE_BUDGETS), N_SEEDS), dtype=float)
    for i, budget in enumerate(SAMPLE_BUDGETS):
        selected = [r.error for r in results if r.method == method and r.n == n and r.budget == budget]
        values[i] = np.asarray(selected, dtype=float)
    return mean_and_ci(values)


def plot_method(results: list[ExperimentResult], method: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(DIMENSIONS)))
    markers = ("o", "s", "D", "^", "P")

    for n, color, marker in zip(DIMENSIONS, colors, markers):
        mean, lo, hi = aggregate_results(results, method, n)
        ax.plot(SAMPLE_BUDGETS, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}")
        ax.fill_between(SAMPLE_BUDGETS, lo, hi, color=color, alpha=0.20)

    ax.axhline(TARGET_ACCURACY, color="red", linestyle="--", linewidth=2.0, label="r=0.01")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of samples N", fontsize=LABEL_SIZE)
    ax.set_ylabel("convex-hull directed Hausdorff error", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = run_experiment()
    save_results_csv(results)
    plot_method(results, "uniform", UNIFORM_FIG, "Uniform Sampling")
    plot_method(results, "adversarial", ADVERSARIAL_FIG, "Adversarial Sampling")

    print("\nSaved:")
    print(UNIFORM_FIG)
    print(ADVERSARIAL_FIG)
    print(CSV_PATH)


if __name__ == "__main__":
    main()
