"""Uniform vs adversarial reachable-set approximation for dx/dt=x^2, dy/dt=0.

Run from /home/jixia/exp with:

    .venv/bin/python reachapprox/exp/quaddynadv/quaddyn_adv_experiment.py

The estimator C(Y_N) used here is the convex hull of endpoint samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Point, Polygon


# Geometry parameters.
R = 1.0
target_area = np.pi
x_c = 2.0
center = np.array([2.0, 0.0])
r_0 = 0.3

# Sampling/optimization parameters.
SAMPLE_BUDGETS = (100, 1000)
TIME_GRID = tuple(float(t) for t in np.geomspace(0.01, 0.28, 13))
TIME_SWEEP_TRIALS = 50
n_adv = 9
eta = 0.018
lambda_reg = 1e-4
RANDOM_SEED = 11

# Numerical parameters.
DISK_RESOLUTION = 512
BUFFER_RESOLUTION = 96
REFERENCE_SAMPLES = 60_000
HULL_CLOUD_SAMPLES = 30_000

# Output paths.
OUT_DIR = Path("reachapprox/exp/quaddynadv")
INITIAL_SETS_FIG = OUT_DIR / "initial_sets.png"
UNIFORM_REACHABLE_FIG = OUT_DIR / "reachable_uniform.png"
ADVERSARIAL_REACHABLE_FIG = OUT_DIR / "reachable_adversarial.png"
HAUSDORFF_FIG = OUT_DIR / "hausdorff_uniform_vs_adversarial.png"
HAUSDORFF_TIME_UNIFORM_FIG = OUT_DIR / "hausdorff_vs_time_uniform.png"
HAUSDORFF_TIME_ADVERSARIAL_FIG = OUT_DIR / "hausdorff_vs_time_adversarial.png"
HAUSDORFF_TIME_CI_FIG = OUT_DIR / "hausdorff_vs_time_uniform_adversarial_ci.png"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)


@dataclass(frozen=True)
class InitialSet:
    name: str
    label: str
    geom: Polygon
    color: str


def flow(points: np.ndarray, T: float) -> np.ndarray:
    """Analytical flow phi(T, (x,y)) = (x/(1-Tx), y)."""
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    denom = 1.0 - T * x
    if np.any(denom <= 0.0):
        raise ValueError("Flow is singular for at least one point.")
    out = points.copy()
    out[:, 0] = x / denom
    return out


def flow_jacobian(points: np.ndarray, T: float) -> np.ndarray:
    """Return D_X phi(T, X) for each point, shape (n, 2, 2)."""
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    diag_x = 1.0 / (1.0 - T * x) ** 2
    jac = np.zeros((points.shape[0], 2, 2), dtype=float)
    jac[:, 0, 0] = diag_x
    jac[:, 1, 1] = 1.0
    return jac


def disk_set() -> Polygon:
    return Point(center).buffer(R, resolution=DISK_RESOLUTION)


def equilateral_triangle() -> Polygon:
    """Equilateral triangle with area pi and centroid exactly at (2,0)."""
    s = 2.0 * np.sqrt(np.pi / np.sqrt(3.0))
    h = np.sqrt(3.0) * s / 2.0
    vertices = np.array(
        [
            [center[0], center[1] + 2.0 * h / 3.0],
            [center[0] - s / 2.0, center[1] - h / 3.0],
            [center[0] + s / 2.0, center[1] - h / 3.0],
        ]
    )
    return Polygon(vertices)


def recenter_to(geom: Polygon, desired_center: np.ndarray = center) -> Polygon:
    c = np.array([geom.centroid.x, geom.centroid.y])
    shift = desired_center - c
    return affinity.translate(geom, xoff=shift[0], yoff=shift[1])


def rescale_area_about_centroid(geom: Polygon, desired_area: float = target_area) -> Polygon:
    scale = np.sqrt(desired_area / geom.area)
    return affinity.scale(geom, xfact=scale, yfact=scale, origin="centroid")


def opened_triangle_set() -> Polygon:
    """Morphological opening (E erosion B_r0) dilation B_r0, then area normalize."""
    tri = equilateral_triangle()
    opened = tri.buffer(-r_0, resolution=BUFFER_RESOLUTION).buffer(r_0, resolution=BUFFER_RESOLUTION)
    opened = rescale_area_about_centroid(opened, target_area)
    opened = recenter_to(opened, center)
    return opened


def min_x(geom: Polygon) -> float:
    xs = [p[0] for p in geom.exterior.coords]
    return float(min(xs))


def max_x(geom: Polygon) -> float:
    xs = [p[0] for p in geom.exterior.coords]
    return float(max(xs))


def check_geometry(initial_sets: list[InitialSet]) -> None:
    for item in initial_sets:
        if item.geom.area <= 0.0:
            raise ValueError(f"{item.name} has non-positive area.")
        if min_x(item.geom) <= 0.0:
            raise ValueError(f"{item.name} is not entirely in x > 0.")


def build_initial_sets() -> list[InitialSet]:
    sets = [
        InitialSet("disk", "Disk", disk_set(), "#c1121f"),
        InitialSet("triangle", "Triangle", equilateral_triangle(), "#1d3557"),
        InitialSet("opened", "Opened triangle", opened_triangle_set(), "#2a9d8f"),
    ]
    check_geometry(sets)
    return sets


def sample_uniform_polygon(rng: np.random.Generator, geom: Polygon, n: int) -> np.ndarray:
    """Uniform rejection sampling from a shapely polygon."""
    minx, miny, maxx, maxy = geom.bounds
    area_box = (maxx - minx) * (maxy - miny)
    accept_rate = max(geom.area / area_box, 1e-3)
    batch = max(1024, int(np.ceil(1.4 * n / accept_rate)))
    path = polygon_path(geom)
    accepted: list[np.ndarray] = []
    count = 0

    while count < n:
        candidates = rng.uniform([minx, miny], [maxx, maxy], size=(batch, 2))
        mask = path.contains_points(candidates, radius=1e-12)
        chosen = candidates[mask]
        if chosen.size:
            accepted.append(chosen)
            count += chosen.shape[0]

    return np.vstack(accepted)[:n]


def project_points_to_set(points: np.ndarray, geom: Polygon) -> np.ndarray:
    """Project points to the closest point in geom; keep interior points fixed."""
    path = polygon_path(geom)
    inside = path.contains_points(points, radius=1e-12)
    projected = points.copy()
    if np.any(~inside):
        projected[~inside] = project_to_polygon_boundary(points[~inside], geom)
    return projected


def polygon_path(geom: Polygon):
    return MplPath(np.asarray(geom.exterior.coords))


def project_to_polygon_boundary(points: np.ndarray, geom: Polygon, chunk_size: int = 512) -> np.ndarray:
    """Vectorized closest-point projection onto a polygon exterior."""
    vertices = np.asarray(geom.exterior.coords[:-1], dtype=float)
    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    segs = ends - starts
    seg_norm2 = np.sum(segs * segs, axis=1)
    seg_norm2[seg_norm2 == 0.0] = 1.0

    out = np.empty_like(points)
    for start_idx in range(0, points.shape[0], chunk_size):
        chunk = points[start_idx : start_idx + chunk_size]
        diff = chunk[:, None, :] - starts[None, :, :]
        tau = np.einsum("nsi,si->ns", diff, segs) / seg_norm2[None, :]
        tau = np.clip(tau, 0.0, 1.0)
        candidates = starts[None, :, :] + tau[:, :, None] * segs[None, :, :]
        dist2 = np.sum((candidates - chunk[:, None, :]) ** 2, axis=2)
        nearest = np.argmin(dist2, axis=1)
        out[start_idx : start_idx + chunk.shape[0]] = candidates[np.arange(chunk.shape[0]), nearest]
    return out


def endpoint_center_and_Q(endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute c and Q using the exact empirical formulas in the prompt."""
    n = endpoints.shape[0]
    c = np.sum(endpoints, axis=0) / n
    centered = endpoints - c
    if n > 1:
        cov = (centered.T @ centered) / (n - 1)
    else:
        cov = np.zeros((2, 2), dtype=float)
    Q = np.linalg.inv(cov + lambda_reg * np.eye(2))
    return c, Q


def adversarial_gradient(points: np.ndarray, T: float, c: np.ndarray, Q: np.ndarray) -> np.ndarray:
    endpoints = flow(points, T)
    diff = endpoints - c
    qdiff = diff @ Q.T
    jac = flow_jacobian(points, T)
    return 2.0 * np.einsum("nij,nj->ni", np.swapaxes(jac, 1, 2), qdiff)


def run_uniform_sampling(
    rng: np.random.Generator,
    geom: Polygon,
    T: float,
    n_samples: int,
) -> tuple[np.ndarray, Polygon]:
    x0 = sample_uniform_polygon(rng, geom, n_samples)
    endpoints = flow(x0, T)
    return endpoints, convex_hull_polygon(endpoints)


def run_adversarial_sampling(
    rng: np.random.Generator,
    geom: Polygon,
    T: float,
    total_budget: int,
) -> tuple[np.ndarray, Polygon]:
    if total_budget % (n_adv + 1) != 0:
        raise ValueError("total_budget must be divisible by n_adv + 1")

    M = total_budget // (n_adv + 1)
    particles = sample_uniform_polygon(rng, geom, M)
    endpoints = [flow(particles, T)]

    for _ in range(n_adv):
        accumulated = np.vstack(endpoints)
        c, Q = endpoint_center_and_Q(accumulated)
        grad = adversarial_gradient(particles, T, c, Q)
        particles = project_points_to_set(particles + eta * grad, geom)
        endpoints.append(flow(particles, T))

    endpoint_cloud = np.vstack(endpoints)
    return endpoint_cloud, convex_hull_polygon(endpoint_cloud)


def convex_hull_polygon(points: np.ndarray) -> Polygon:
    hull = MultiPoint(points).convex_hull
    if hull.geom_type == "Polygon":
        return hull
    return hull.buffer(1e-10)


def sample_from_convex_hull(rng: np.random.Generator, hull: Polygon, n: int) -> np.ndarray:
    coords = np.asarray(hull.exterior.coords[:-1], dtype=float)
    if coords.shape[0] < 3:
        return np.repeat(coords[:1], n, axis=0)

    anchor = coords.mean(axis=0)
    starts = coords
    ends = np.roll(coords, -1, axis=0)
    cross = np.abs(np.cross(starts - anchor, ends - anchor))
    areas = 0.5 * cross
    if float(np.sum(areas)) <= 0.0:
        return sample_uniform_polygon(rng, hull, n)

    tri_indices = rng.choice(len(areas), size=n, p=areas / np.sum(areas))
    a = np.repeat(anchor[None, :], n, axis=0)
    b = starts[tri_indices]
    c = ends[tri_indices]
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return a + u[:, None] * (b - a) + v[:, None] * (c - a)


def approximate_hausdorff(
    rng: np.random.Generator,
    reference_points: np.ndarray,
    estimate_hull: Polygon,
    n_hull_samples: int = HULL_CLOUD_SAMPLES,
) -> float:
    estimate_points = sample_from_convex_hull(rng, estimate_hull, n_hull_samples)
    tree_ref = cKDTree(reference_points)
    tree_est = cKDTree(estimate_points)
    ref_to_est = tree_est.query(reference_points, k=1)[0].max()
    est_to_ref = tree_ref.query(estimate_points, k=1)[0].max()
    return float(max(ref_to_est, est_to_ref))


def boundary_points(geom: Polygon, n_points: int) -> np.ndarray:
    """Evenly sample points on the exterior boundary of a polygon."""
    ring = LineString(geom.exterior.coords)
    distances = np.linspace(0.0, ring.length, n_points, endpoint=False)
    points = [ring.interpolate(float(distance)) for distance in distances]
    return np.array([[point.x, point.y] for point in points], dtype=float)


def approximate_boundary_hausdorff(reference_boundary: np.ndarray, estimate_hull: Polygon) -> float:
    """Approximate Hausdorff distance using true and estimated boundary point clouds."""
    estimate_boundary = boundary_points(estimate_hull, 2_000)
    tree_ref = cKDTree(reference_boundary)
    tree_est = cKDTree(estimate_boundary)
    ref_to_est = tree_est.query(reference_boundary, k=1)[0].max()
    est_to_ref = tree_ref.query(estimate_boundary, k=1)[0].max()
    return float(max(ref_to_est, est_to_ref))


def transformed_boundary(geom: Polygon, T: float, n_per_edge: int = 5) -> np.ndarray:
    coords = np.asarray(geom.exterior.coords)
    pieces = []
    for start, end in zip(coords[:-1], coords[1:]):
        w = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
        pieces.append((1.0 - w[:, None]) * start + w[:, None] * end)
    return flow(np.vstack(pieces), T)


def polygon_patch(geom: Polygon, **kwargs) -> MplPolygon:
    coords = np.asarray(geom.exterior.coords)
    return MplPolygon(coords, closed=True, **kwargs)


def plot_initial_sets(initial_sets: list[InitialSet]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    for item in initial_sets:
        ax.add_patch(polygon_patch(item.geom, facecolor=item.color, alpha=0.24, edgecolor=item.color, lw=2.0))
        centroid = item.geom.centroid
        print(
            f"{item.label:<16} area={item.geom.area:.8f}, "
            f"centroid=({centroid.x:.8f}, {centroid.y:.8f}), min_x={min_x(item.geom):.8f}"
        )

    ax.scatter([center[0]], [center[1]], color="black", s=42, zorder=5, label="common center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Equal-area initial sets")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(INITIAL_SETS_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reachable_clouds(
    results: dict[tuple[str, str, int], tuple[np.ndarray, Polygon]],
    initial_sets: list[InitialSet],
    T: float,
    method: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    budget = max(SAMPLE_BUDGETS)
    for ax, item in zip(axes, initial_sets):
        endpoints, hull = results[(item.name, method, budget)]
        boundary = transformed_boundary(item.geom, T)
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=3, color=item.color, alpha=0.28, linewidths=0)
        ax.add_patch(polygon_patch(hull, facecolor=item.color, alpha=0.12, edgecolor=item.color, lw=2.0))
        ax.plot(boundary[:, 0], boundary[:, 1], color="black", lw=1.2)
        ax.set_title(f"{item.label}, N={budget}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("x(T)")
    axes[0].set_ylabel("y(T)")
    fig.suptitle(f"{method.capitalize()} sampling reachable-set estimates")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hausdorff(errors: dict[tuple[str, str, int], float], initial_sets: list[InitialSet]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    for item in initial_sets:
        for method, linestyle, marker in (("uniform", "-", "o"), ("adversarial", "--", "s")):
            ys = [errors[(item.name, method, n)] for n in SAMPLE_BUDGETS]
            ax.plot(
                SAMPLE_BUDGETS,
                ys,
                linestyle=linestyle,
                marker=marker,
                color=item.color,
                lw=2.0,
                ms=6,
                label=f"{item.label}, {method}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample budget N")
    ax.set_ylabel("Hausdorff distance")
    ax.set_title("Convex-hull reachable-set error")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(HAUSDORFF_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_time_sweep(
    initial_sets: list[InitialSet],
) -> dict[tuple[str, str, float, int], float]:
    """Compute Hausdorff errors over TIME_GRID for uniform/adversarial sampling."""
    rng = np.random.default_rng(RANDOM_SEED + 40_000)
    reference_initial = {
        item.name: sample_uniform_polygon(rng, item.geom, REFERENCE_SAMPLES)
        for item in initial_sets
    }
    errors: dict[tuple[str, str, float, int], float] = {}

    print("\nRunning Hausdorff-vs-time sweep:")
    for item in initial_sets:
        for t in TIME_GRID:
            reference_t = flow(reference_initial[item.name], t)
            for budget in SAMPLE_BUDGETS:
                uniform_rng = np.random.default_rng(
                    RANDOM_SEED + 50_000 + stable_seed_offset(item.name, "uniform", t, budget)
                )
                adv_rng = np.random.default_rng(
                    RANDOM_SEED + 60_000 + stable_seed_offset(item.name, "adversarial", t, budget)
                )
                err_rng = np.random.default_rng(
                    RANDOM_SEED + 70_000 + stable_seed_offset(item.name, "error", t, budget)
                )

                _, uniform_hull = run_uniform_sampling(uniform_rng, item.geom, t, budget)
                _, adv_hull = run_adversarial_sampling(adv_rng, item.geom, t, budget)

                errors[(item.name, "uniform", t, budget)] = approximate_hausdorff(
                    err_rng, reference_t, uniform_hull
                )
                errors[(item.name, "adversarial", t, budget)] = approximate_hausdorff(
                    err_rng, reference_t, adv_hull
                )

                print(
                    f"{item.label:<16} t={t:.4f} N={budget:<4d} "
                    f"uniform={errors[(item.name, 'uniform', t, budget)]:>10.6f} "
                    f"adversarial={errors[(item.name, 'adversarial', t, budget)]:>10.6f}"
                )
    return errors


def stable_seed_offset(set_name: str, method: str, t: float, budget: int) -> int:
    """Small deterministic offset that is stable across Python processes."""
    text = f"{set_name}:{method}:{t:.8f}:{budget}"
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def plot_hausdorff_vs_time(
    errors: dict[tuple[str, str, float, int], float],
    initial_sets: list[InitialSet],
    method: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), constrained_layout=True)
    sample_colors = {
        100: "tab:orange",
        1000: "tab:green",
    }
    markers = {
        100: "s",
        1000: "D",
    }

    for ax, item in zip(axes, initial_sets):
        for budget in SAMPLE_BUDGETS:
            ys = [errors[(item.name, method, t, budget)] for t in TIME_GRID]
            ax.plot(
                TIME_GRID,
                ys,
                color=sample_colors[budget],
                marker=markers[budget],
                lw=2.0,
                ms=5.0,
                label=f"N={budget}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(TIME_GRID)
        ax.set_xticklabels([f"{t:.4f}" for t in TIME_GRID], rotation=35, ha="right", fontsize=8)
        ax.set_title(item.label)
        ax.set_xlabel("time t")
        ax.grid(True, which="both", alpha=0.28)

    axes[0].set_ylabel("Hausdorff distance")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(f"Reachable-set approximation error ({method} sampling)")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_time_sweep_ci(
    initial_sets: list[InitialSet],
) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run 50-seed Hausdorff-vs-time comparison and return mean/95% CI curves."""
    curves: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    print(f"\nRunning {TIME_SWEEP_TRIALS}-seed Hausdorff-vs-time CI sweep:")

    for item in initial_sets:
        reference_boundaries = {
            t: flow(boundary_points(item.geom, 4_000), t)
            for t in TIME_GRID
        }
        for budget in SAMPLE_BUDGETS:
            values = {
                "uniform": np.empty((len(TIME_GRID), TIME_SWEEP_TRIALS), dtype=float),
                "adversarial": np.empty((len(TIME_GRID), TIME_SWEEP_TRIALS), dtype=float),
            }
            for t_index, t in enumerate(TIME_GRID):
                reference_boundary = reference_boundaries[t]
                for trial in range(TIME_SWEEP_TRIALS):
                    uniform_rng = np.random.default_rng(
                        RANDOM_SEED
                        + 80_000
                        + stable_seed_offset(item.name, "uniform-ci", t, budget)
                        + trial
                    )
                    adv_rng = np.random.default_rng(
                        RANDOM_SEED
                        + 90_000
                        + stable_seed_offset(item.name, "adv-ci", t, budget)
                        + trial
                    )
                    _, uniform_hull = run_uniform_sampling(uniform_rng, item.geom, t, budget)
                    _, adv_hull = run_adversarial_sampling(adv_rng, item.geom, t, budget)
                    values["uniform"][t_index, trial] = approximate_boundary_hausdorff(
                        reference_boundary, uniform_hull
                    )
                    values["adversarial"][t_index, trial] = approximate_boundary_hausdorff(
                        reference_boundary, adv_hull
                    )

                print(
                    f"{item.label:<16} t={t:.4f} N={budget:<4d} "
                    f"uniform mean={values['uniform'][t_index].mean():>10.6f} "
                    f"adv mean={values['adversarial'][t_index].mean():>10.6f}"
                )

            for method in ("uniform", "adversarial"):
                mean = values[method].mean(axis=1)
                stderr = values[method].std(axis=1, ddof=1) / np.sqrt(TIME_SWEEP_TRIALS)
                ci = 1.96 * stderr
                lo = np.maximum(mean - ci, np.finfo(float).tiny)
                hi = mean + ci
                curves[(item.name, method, budget)] = (mean, lo, hi)

    return curves


def plot_hausdorff_vs_time_ci(
    curves: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    initial_sets: list[InitialSet],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)
    sample_colors = {
        100: "tab:orange",
        1000: "tab:green",
    }
    method_styles = {
        "uniform": ("-", "o", "uniform"),
        "adversarial": ("--", "s", "adversarial"),
    }

    for ax, item in zip(axes, initial_sets):
        for budget in SAMPLE_BUDGETS:
            color = sample_colors[budget]
            for method, (linestyle, marker, label) in method_styles.items():
                mean, lo, hi = curves[(item.name, method, budget)]
                ax.plot(
                    TIME_GRID,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    lw=2.0,
                    ms=4.6,
                    label=f"N={budget}, {label}",
                )
                ax.fill_between(TIME_GRID, lo, hi, color=color, alpha=0.22)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(TIME_GRID)
        ax.set_xticklabels([f"{t:.4f}" for t in TIME_GRID], rotation=35, ha="right", fontsize=8)
        ax.set_title(item.label)
        ax.set_xlabel("time t")
        ax.grid(True, which="both", alpha=0.28)

    axes[0].set_ylabel("Hausdorff distance")
    axes[-1].legend(frameon=False, fontsize=7.4, loc="best")
    fig.suptitle("Reachable-set approximation error")
    fig.savefig(HAUSDORFF_TIME_CI_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    initial_sets = build_initial_sets()

    max_initial_x = max(max_x(item.geom) for item in initial_sets)
    T = 0.5 / max_initial_x
    print(f"Chosen T = {T:.8f}; max_x = {max_initial_x:.8f}; T*max_x = {T * max_initial_x:.4f}")
    print("\nInitial-set checks:")
    plot_initial_sets(initial_sets)

    reference_points: dict[str, np.ndarray] = {}
    for item in initial_sets:
        ref_initial = sample_uniform_polygon(rng, item.geom, REFERENCE_SAMPLES)
        reference_points[item.name] = flow(ref_initial, T)

    results: dict[tuple[str, str, int], tuple[np.ndarray, Polygon]] = {}
    errors: dict[tuple[str, str, int], float] = {}

    print("\nRunning sampling comparison:")
    for item in initial_sets:
        for budget in SAMPLE_BUDGETS:
            uniform_rng = np.random.default_rng(RANDOM_SEED + 1000 + budget)
            adv_rng = np.random.default_rng(RANDOM_SEED + 2000 + budget)
            err_rng = np.random.default_rng(RANDOM_SEED + 3000 + budget)

            uniform_points, uniform_hull = run_uniform_sampling(uniform_rng, item.geom, T, budget)
            adv_points, adv_hull = run_adversarial_sampling(adv_rng, item.geom, T, budget)

            results[(item.name, "uniform", budget)] = (uniform_points, uniform_hull)
            results[(item.name, "adversarial", budget)] = (adv_points, adv_hull)

            errors[(item.name, "uniform", budget)] = approximate_hausdorff(
                err_rng, reference_points[item.name], uniform_hull
            )
            errors[(item.name, "adversarial", budget)] = approximate_hausdorff(
                err_rng, reference_points[item.name], adv_hull
            )

            print(
                f"{item.label:<16} N={budget:<5d} "
                f"uniform={errors[(item.name, 'uniform', budget)]:>10.6f} "
                f"adversarial={errors[(item.name, 'adversarial', budget)]:>10.6f}"
            )

    plot_reachable_clouds(results, initial_sets, T, "uniform", UNIFORM_REACHABLE_FIG)
    plot_reachable_clouds(results, initial_sets, T, "adversarial", ADVERSARIAL_REACHABLE_FIG)
    plot_hausdorff(errors, initial_sets)
    time_curves = run_time_sweep_ci(initial_sets)
    plot_hausdorff_vs_time_ci(time_curves, initial_sets)

    print("\nSaved figures:")
    for path in (
        INITIAL_SETS_FIG,
        UNIFORM_REACHABLE_FIG,
        ADVERSARIAL_REACHABLE_FIG,
        HAUSDORFF_FIG,
        HAUSDORFF_TIME_CI_FIG,
    ):
        print(path)


if __name__ == "__main__":
    main()
