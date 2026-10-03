"""Figure 6 (Appendix C.3): role of the uniform density lower bound.

The support, dynamics, and horizon are fixed: the unit disk centered at
(1, 0), x' = 2x, y' = 0, T = 0.5, so the reachable set is always the ellipse
(X - e^{2T})^2 / e^{4T} + Y^2 <= 1.  Only the sampling density changes,

    p_beta(x, y) = ((beta + 1)(beta + 2) / (2 pi)) (1 - r)^beta,   beta in {0, 2, 4},

which vanishes at the boundary for beta > 0 (beta = 0 is uniform).  Samples
are exact: r ~ Beta(2, beta + 1), angle uniform.  Convex-hull, fixed-radius
union-of-balls (h = 0.05) and Christoffel estimators are compared over 50
seeds; the curves are the mean symmetric Hausdorff error with 5-95% bands.

    python -m experiments.fig6_density_lower_bound
    python -m experiments.fig6_density_lower_bound --plot-only
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import shapely
from shapely.geometry import MultiPoint

from reachapprox.estimators import monomial_powers, polynomial_features
from reachapprox.flows import linear_flow
from reachapprox.utils import RESULTS_DIR, parallel_map, read_csv, use_serif_fonts, write_csv


BETAS = (0, 2, 4)
SAMPLE_BUDGETS = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
N_SEEDS = 50
BASE_SEED = 20260713
FINAL_TIME = 0.5
GROWTH_RATE = 2.0
SUPPORT_CENTER = np.array([1.0, 0.0])
PACKING_RADIUS = 0.05
CHRISTOFFEL_DEGREE = 6
CHRISTOFFEL_REGULARIZATION = 1e-6
GRID_RESOLUTION = 120
GRID_PADDING_FRACTION = 0.08
REFERENCE_RADIAL_LEVELS = 64
REFERENCE_ANGLES = 720
ELLIPSE_BOUNDARY_POINTS = 4_000
SUPPORT_FUNCTION_ANGLES = 4_096

ESTIMATORS = ("convex_hull", "packing", "christoffel")
TITLES = {"convex_hull": "(a) Convex hull", "packing": "(b) Packing-ball union", "christoffel": "(c) Christoffel function"}
COLORS = {0: "#2a6fbb", 2: "#e07a1f", 4: "#c83e4d"}
LINESTYLES = {0: "-", 2: "--", 4: "-."}
LABELS = {0: r"$\beta=0$ (uniform)", 2: r"$\beta=2$", 4: r"$\beta=4$"}

OUT_DIR = RESULTS_DIR / "fig6"
RAW_CSV = OUT_DIR / "fig6_trials.csv"
FIGURE_PATH = OUT_DIR / "fig6_density_lower_bound.png"

ELLIPSE_AXES = (np.exp(GROWTH_RATE * FINAL_TIME), 1.0)


def sample_disk_density(rng: np.random.Generator, n_samples: int, beta: int) -> np.ndarray:
    """Exact samples from p_beta on the unit disk centered at (1, 0)."""
    radii = rng.beta(2.0, beta + 1.0, size=n_samples)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    return SUPPORT_CENTER + np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def radial_density(radius: np.ndarray, beta: int) -> np.ndarray:
    """Density of r = |x - center| under p_beta: (beta+1)(beta+2) r (1-r)^beta."""
    radius = np.asarray(radius, dtype=float)
    values = (beta + 1) * (beta + 2) * radius * np.maximum(1.0 - radius, 0.0) ** beta
    return np.where((radius >= 0.0) & (radius <= 1.0), values, 0.0)


def flow(points: np.ndarray) -> np.ndarray:
    return linear_flow(points, FINAL_TIME, rate=GROWTH_RATE)


def terminal_center() -> np.ndarray:
    return flow(SUPPORT_CENTER[None, :])[0]


def ellipse_boundary(n_points: int = ELLIPSE_BOUNDARY_POINTS) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    a, b = ELLIPSE_AXES
    return terminal_center() + np.column_stack((a * np.cos(theta), b * np.sin(theta)))


def ellipse_reference() -> np.ndarray:
    """Deterministic discretization of the filled terminal ellipse."""
    radii = np.linspace(0.0, 1.0, REFERENCE_RADIAL_LEVELS + 1)[1:]
    theta = np.linspace(0.0, 2.0 * np.pi, REFERENCE_ANGLES, endpoint=False)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    a, b = ELLIPSE_AXES
    center = terminal_center()
    interior = center + np.column_stack((a * rr.ravel() * np.cos(tt).ravel(), b * rr.ravel() * np.sin(tt).ravel()))
    return np.vstack((center[None, :], interior, ellipse_boundary()))


def evaluation_grid() -> np.ndarray:
    a, b = ELLIPSE_AXES
    center = terminal_center()
    pad_x, pad_y = 2.0 * a * GRID_PADDING_FRACTION, 2.0 * b * GRID_PADDING_FRACTION
    xs = np.linspace(center[0] - a - pad_x, center[0] + a + pad_x, GRID_RESOLUTION)
    ys = np.linspace(center[1] - b - pad_y, center[1] + b + pad_y, GRID_RESOLUTION)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def distance_to_ellipse(points: np.ndarray) -> np.ndarray:
    """Euclidean distance to the filled terminal ellipse (bisection on the Lagrange multiplier)."""
    points = np.asarray(points, dtype=float) - terminal_center()
    a, b = ELLIPSE_AXES
    x, y = np.abs(points[:, 0]), np.abs(points[:, 1])
    distances = np.zeros(points.shape[0], dtype=float)
    outside = (x / a) ** 2 + (y / b) ** 2 > 1.0
    if not np.any(outside):
        return distances

    xo, yo = x[outside], y[outside]
    a2, b2 = a * a, b * b

    def equation(lam: np.ndarray) -> np.ndarray:
        return (a * xo / (lam + a2)) ** 2 + (b * yo / (lam + b2)) ** 2 - 1.0

    lower = np.zeros_like(xo)
    upper = np.maximum(np.hypot(xo, yo) * max(a, b), 1.0)
    while np.any(equation(upper) > 0.0):
        upper[equation(upper) > 0.0] *= 2.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        positive = equation(middle) > 0.0
        lower[positive] = middle[positive]
        upper[~positive] = middle[~positive]
    lam = 0.5 * (lower + upper)
    distances[outside] = np.hypot(xo - a2 * xo / (lam + a2), yo - b2 * yo / (lam + b2))
    return distances


def greedy_maximal_packing(samples: np.ndarray, radius: float) -> np.ndarray:
    """Greedy maximal radius-separated subset of the samples (spatial hash)."""
    buckets: dict[tuple[int, int], list[int]] = {}
    centers: list[np.ndarray] = []
    for sample in np.asarray(samples, dtype=float):
        cell = tuple(np.floor(sample / radius).astype(int))
        neighbors = (
            index
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for index in buckets.get((cell[0] + dx, cell[1] + dy), ())
        )
        if all(np.linalg.norm(sample - centers[index]) > radius for index in neighbors):
            buckets.setdefault(cell, []).append(len(centers))
            centers.append(sample.copy())
    return np.asarray(centers, dtype=float)


def packing_outer_error(centers: np.ndarray, radius: float) -> float:
    """sup over the union of balls of the distance to the ellipse, via support functions."""
    support_centers = centers
    if centers.shape[0] >= 3:
        try:
            support_centers = centers[ConvexHull(centers).vertices]
        except Exception:
            pass
    theta = np.linspace(0.0, 2.0 * np.pi, SUPPORT_FUNCTION_ANGLES, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    center_support = np.full(directions.shape[0], -np.inf)
    for start in range(0, support_centers.shape[0], 1024):
        center_support = np.maximum(center_support, np.max(support_centers[start : start + 1024] @ directions.T, axis=0))
    a, b = ELLIPSE_AXES
    ellipse_support = terminal_center() @ directions.T + np.sqrt((a * directions[:, 0]) ** 2 + (b * directions[:, 1]) ** 2)
    return float(max(0.0, np.max(center_support + radius - ellipse_support)))


def christoffel_mask(samples: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    """Christoffel sublevel set {kappa_N <= max_i kappa_N(Y_i)} on the grid (linear solves)."""
    lower, upper = grid_points.min(axis=0), grid_points.max(axis=0)
    center, scale = 0.5 * (lower + upper), 0.5 * (upper - lower)
    powers = monomial_powers(CHRISTOFFEL_DEGREE)
    phi_samples = polynomial_features((samples - center) / scale, powers)
    gram = (phi_samples.T @ phi_samples) / samples.shape[0]
    ridge = CHRISTOFFEL_REGULARIZATION * max(float(np.trace(gram)) / gram.shape[0], 1.0)
    matrix = gram + ridge * np.eye(gram.shape[0])
    sample_values = np.einsum("ij,ij->i", phi_samples, np.linalg.solve(matrix, phi_samples.T).T)
    threshold = float(np.max(sample_values)) * (1.0 + 1e-10)
    phi_grid = polynomial_features((grid_points - center) / scale, powers)
    mask = np.einsum("ij,ij->i", phi_grid, np.linalg.solve(matrix, phi_grid.T).T) <= threshold
    if not np.any(mask):
        mask[int(cKDTree(grid_points).query(samples[:1], k=1)[1][0])] = True
    return mask


def directed_errors(estimator: str, samples: np.ndarray, reference: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    """Inner sup_{x in S_T} d(x, S_N) and outer sup_{z in S_N} d(z, S_T) errors."""
    if estimator == "convex_hull":
        hull = MultiPoint(samples).convex_hull
        inner = float(np.max(shapely.distance(shapely.points(reference[:, 0], reference[:, 1]), hull)))
        return inner, 0.0
    if estimator == "packing":
        centers = greedy_maximal_packing(samples, PACKING_RADIUS)
        nearest = cKDTree(centers).query(reference, k=1)[0]
        return float(np.max(np.maximum(nearest - PACKING_RADIUS, 0.0))), packing_outer_error(centers, PACKING_RADIUS)
    if estimator == "christoffel":
        estimate = np.vstack((grid[christoffel_mask(samples, grid)], samples))
        inner = float(cKDTree(estimate).query(reference, k=1)[0].max())
        return inner, float(distance_to_ellipse(estimate).max())
    raise ValueError(f"unknown estimator: {estimator}")


def run_condition(task: tuple[int, int, int, int]) -> list[dict]:
    """All seeds and estimators of one (budget, beta) condition."""
    budget_index, n_samples, beta, n_seeds = task
    reference = ellipse_reference()
    grid = evaluation_grid()
    rows = []
    for seed_index in range(n_seeds):
        seed = BASE_SEED + 10_000 * budget_index + seed_index
        endpoints = flow(sample_disk_density(np.random.default_rng(seed), n_samples, beta))
        for estimator in ESTIMATORS:
            inner, outer = directed_errors(estimator, endpoints, reference, grid)
            rows.append({"beta": beta, "N": n_samples, "seed_index": seed_index, "seed": seed,
                         "estimator": estimator, "hausdorff": max(inner, outer),
                         "inner_directed": inner, "outer_directed": outer})
    return rows


def run_experiment(budgets: tuple[int, ...], n_seeds: int, workers: int | None) -> list[dict]:
    tasks = [(index, n_samples, beta, n_seeds) for index, n_samples in enumerate(budgets) for beta in BETAS]
    rows = []
    for (_, n_samples, beta, _), task_rows in zip(tasks, parallel_map(run_condition, tasks, workers)):
        mean = np.mean([row["hausdorff"] for row in task_rows if row["estimator"] == "christoffel"])
        print(f"N={n_samples:>7} beta={beta}: Christoffel mean d_H={mean:.5f}")
        rows += task_rows
    return rows


def plot_figure(rows: list[dict]) -> None:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["beta"]), int(row["N"]), row["estimator"]), []).append(float(row["hausdorff"]))
    budgets = sorted({key[1] for key in grouped})

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True, sharey=True)
    for ax, estimator in zip(axes, ESTIMATORS):
        for beta in BETAS:
            values = [np.asarray(grouped[(beta, n, estimator)]) for n in budgets]
            ax.plot(budgets, [v.mean() for v in values], color=COLORS[beta], linestyle=LINESTYLES[beta],
                    marker="o", ms=4, lw=1.8, label=LABELS[beta])
            ax.fill_between(budgets, [np.quantile(v, 0.05) for v in values], [np.quantile(v, 0.95) for v in values],
                            color=COLORS[beta], alpha=0.14, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"Sample size $N$")
        ax.set_title(TITLES[estimator])
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"Hausdorff error $d_H$")
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.savefig(FIGURE_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print("\nMean Christoffel error at the largest budget:")
    for beta in BETAS:
        print(f"  beta={beta}: {np.mean(grouped[(beta, budgets[-1], 'christoffel')]):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="re-plot from the saved trials CSV")
    parser.add_argument("--budgets", default=",".join(map(str, SAMPLE_BUDGETS)), help="comma-separated N values")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    args = parser.parse_args()
    use_serif_fonts("DejaVu Serif")

    if args.plot_only:
        rows = read_csv(RAW_CSV)
    else:
        rows = run_experiment(tuple(int(n) for n in args.budgets.split(",")), args.seeds, args.workers)
        write_csv(rows, RAW_CSV)
        print(f"saved {RAW_CSV}")
    plot_figure(rows)
    print(f"saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
