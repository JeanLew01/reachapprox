"""Reachable-set approximation experiment for dx/dt = x^2, dy/dt = 0.

Run from the repository root with:

    python reachapprox/hausdorff_experiment.py

The script prints the mean approximate Hausdorff distances and saves
``hausdorff_vs_samples.png`` in the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import cKDTree

LABEL_SIZE = 28
TITLE_SIZE = 28
TICK_SIZE = 22
LEGEND_SIZE = 22
SCHEMATIC_LABEL_SIZE = 42
SCHEMATIC_TITLE_SIZE = 50
SCHEMATIC_TICK_SIZE = 34
SCHEMATIC_LEGEND_SIZE = 42

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
    }
)

PLOT_FONT = {"fontname": "DejaVu Serif"}
PLOT_FONT_PROP = {"family": "DejaVu Serif", "size": LEGEND_SIZE}
SCHEMATIC_TITLE_FONT = {"fontname": "DejaVu Serif", "fontsize": SCHEMATIC_TITLE_SIZE}
SCHEMATIC_LABEL_FONT = {"fontname": "DejaVu Serif", "fontsize": SCHEMATIC_LABEL_SIZE}
SCHEMATIC_FONT_PROP = {"family": "DejaVu Serif", "size": SCHEMATIC_LEGEND_SIZE}


CENTER = np.array([2.0, 0.0])
OUTER_RADIUS = 1.0
INNER_RADIUS = OUTER_RADIUS * np.sin(np.pi / 10.0) / np.sin(3.0 * np.pi / 10.0)

TIMES = tuple(float(t) for t in np.geomspace(0.01, 0.33, 7))
SAMPLE_SIZES = (10, 100, 1000)
N_TRIALS = 20

POLY_DEGREE = 6
REGULARIZATION = 1e-6
GRID_RESOLUTION = 170
N_TRUE_POINTS = 35_000
N_SCHEMATIC_SAMPLES = 200
RANDOM_SEED = 7

FIGURE_PATH = Path("hausdorff_vs_samples.png")
SAMPLE_FLOW_FIGURE_PATH = Path("sample_flow_schematic.png")


@dataclass(frozen=True)
class InitialSet:
    name: str
    label: str
    polygon: np.ndarray | None = None


def flow(points: np.ndarray, t: float) -> np.ndarray:
    """Analytical flow T_t(x0, y0) = (x0 / (1 - t x0), y0)."""
    points = np.asarray(points, dtype=float)
    x0 = points[:, 0]
    denominator = 1.0 - t * x0
    if np.any(denominator <= 0.0):
        raise ValueError("flow is singular for at least one point: 1 - t*x0 <= 0")

    out = points.copy()
    out[:, 0] = x0 / denominator
    return out


def star_vertices(center: np.ndarray = CENTER, outer_radius: float = OUTER_RADIUS) -> np.ndarray:
    """Return a 10-vertex regular five-pointed star with right outer vertex."""
    angles = np.arange(10) * np.pi / 5.0
    radii = np.where(np.arange(10) % 2 == 0, outer_radius, INNER_RADIUS)
    vertices = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return vertices + center


def disk_boundary(n_points: int = 1_000) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    offsets = OUTER_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))
    return CENTER + offsets


def polygon_boundary(vertices: np.ndarray, points_per_edge: int = 120) -> np.ndarray:
    pieces = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        weights = np.linspace(0.0, 1.0, points_per_edge, endpoint=False)
        pieces.append((1.0 - weights[:, None]) * start + weights[:, None] * end)
    return np.vstack(pieces)


def sample_disk(rng: np.random.Generator, n: int) -> np.ndarray:
    radius = OUTER_RADIUS * np.sqrt(rng.random(n))
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    offsets = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    return CENTER + offsets


def sample_polygon_rejection(
    rng: np.random.Generator,
    polygon: np.ndarray,
    n: int,
    batch_size: int | None = None,
) -> np.ndarray:
    """Uniformly sample a polygon using rejection from its bounding box."""
    path = MplPath(polygon)
    lower = polygon.min(axis=0)
    upper = polygon.max(axis=0)
    batch_size = max(4 * n, 1_000) if batch_size is None else batch_size

    accepted: list[np.ndarray] = []
    count = 0
    while count < n:
        candidates = rng.uniform(lower, upper, size=(batch_size, 2))
        inside = path.contains_points(candidates, radius=1e-12)
        chosen = candidates[inside]
        if chosen.size:
            accepted.append(chosen)
            count += chosen.shape[0]

    return np.vstack(accepted)[:n]


def sample_initial_set(rng: np.random.Generator, initial_set: InitialSet, n: int) -> np.ndarray:
    if initial_set.name == "disk":
        return sample_disk(rng, n)
    if initial_set.name == "star" and initial_set.polygon is not None:
        return sample_polygon_rejection(rng, initial_set.polygon, n)
    raise ValueError(f"unknown initial set {initial_set.name!r}")


def monomial_powers(degree: int) -> list[tuple[int, int]]:
    return [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]


def polynomial_features(points: np.ndarray, powers: list[tuple[int, int]]) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    features = np.empty((points.shape[0], len(powers)), dtype=float)
    for k, (px, py) in enumerate(powers):
        features[:, k] = (x**px) * (y**py)
    return features


def christoffel_estimator_mask(
    samples: np.ndarray,
    grid_points: np.ndarray,
    degree: int = POLY_DEGREE,
    regularization: float = REGULARIZATION,
) -> np.ndarray:
    """Classify grid points using an empirical Christoffel sublevel set."""
    lower = grid_points.min(axis=0)
    upper = grid_points.max(axis=0)
    center = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    scale[scale == 0.0] = 1.0

    samples_scaled = (samples - center) / scale
    grid_scaled = (grid_points - center) / scale

    powers = monomial_powers(degree)
    phi_samples = polynomial_features(samples_scaled, powers)
    gram = (phi_samples.T @ phi_samples) / samples.shape[0]
    ridge = regularization * max(float(np.trace(gram)) / gram.shape[0], 1.0)
    gram = gram + ridge * np.eye(gram.shape[0])

    inv_gram = np.linalg.pinv(gram, hermitian=True)
    k_samples = np.einsum("ij,jk,ik->i", phi_samples, inv_gram, phi_samples)
    threshold = float(np.max(k_samples)) * (1.0 + 1e-10)

    phi_grid = polynomial_features(grid_scaled, powers)
    k_grid = np.einsum("ij,jk,ik->i", phi_grid, inv_gram, phi_grid)
    mask = k_grid <= threshold

    if not np.any(mask):
        distances, indices = cKDTree(grid_points).query(samples, k=1)
        nearest_sample = int(np.argmin(distances))
        mask[int(indices[nearest_sample])] = True

    return mask


def make_grid(true_points: np.ndarray, samples: np.ndarray, resolution: int) -> np.ndarray:
    all_points = np.vstack((true_points, samples))
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    span = upper - lower
    padding = 0.08 * np.maximum(span, 1e-6)
    lower = lower - padding
    upper = upper + padding

    xs = np.linspace(lower[0], upper[0], resolution)
    ys = np.linspace(lower[1], upper[1], resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def approximate_hausdorff(true_points: np.ndarray, estimated_points: np.ndarray) -> float:
    """Approximate symmetric Hausdorff distance between two point clouds."""
    if estimated_points.shape[0] == 0:
        return float("inf")

    tree_true = cKDTree(true_points)
    tree_est = cKDTree(estimated_points)
    true_to_est = tree_est.query(true_points, k=1)[0].max()
    est_to_true = tree_true.query(estimated_points, k=1)[0].max()
    return float(max(true_to_est, est_to_true))


def run_trial(
    rng: np.random.Generator,
    initial_set: InitialSet,
    t: float,
    n_samples: int,
    true_points_t: np.ndarray,
) -> float:
    initial_samples = sample_initial_set(rng, initial_set, n_samples)
    endpoint_samples = flow(initial_samples, t)
    grid_points = make_grid(true_points_t, endpoint_samples, GRID_RESOLUTION)
    mask = christoffel_estimator_mask(endpoint_samples, grid_points)
    estimated_points = grid_points[mask]
    return approximate_hausdorff(true_points_t, estimated_points)


def print_table(results: dict[tuple[str, float, int], float]) -> None:
    header = f"{'set':<8} {'t':>5} {'N':>6} {'mean Hausdorff':>18}"
    print(header)
    print("-" * len(header))
    for set_name in ("disk", "star"):
        for t in TIMES:
            for n in SAMPLE_SIZES:
                value = results[(set_name, t, n)]
                print(f"{set_name:<8} {t:>8.4f} {n:>6d} {value:>18.6f}")


def plot_results(results: dict[tuple[str, float, int], float]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 9.735), constrained_layout=True)
    plot_times = np.array(TIMES, dtype=float)

    sample_colors = {
        10: "tab:blue",
        100: "tab:orange",
        1000: "tab:green",
    }
    styles = {
        ("disk", 10): ("o", "-", "S1 disk, N = 10"),
        ("disk", 100): ("s", "-", "S1 disk, N = 100"),
        ("disk", 1000): ("D", "-", "S1 disk, N = 1000"),
        ("star", 10): ("^", "--", "S2 star, N = 10"),
        ("star", 100): ("v", "--", "S2 star, N = 100"),
        ("star", 1000): ("P", "--", "S2 star, N = 1000"),
    }

    for key, (marker, linestyle, label) in styles.items():
        set_name, n = key
        ys = [results[(set_name, t, n)] for t in TIMES]
        ax.plot(
            plot_times,
            ys,
            color=sample_colors[n],
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,
            markersize=6.0,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time t", **PLOT_FONT)
    ax.set_ylabel("Hausdorff distance", **PLOT_FONT)
    ax.set_title("Reachable-set approximation error", **PLOT_FONT)
    ax.set_xticks(plot_times)
    ax.set_xticklabels([f"{t:.4f}".rstrip("0").rstrip(".") for t in TIMES])
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, prop=PLOT_FONT_PROP)
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_sample_flow_schematic() -> None:
    """Save a schematic of disk/star samples and exact transported boundaries."""
    rng = np.random.default_rng(RANDOM_SEED)
    star = InitialSet("star", "S2 star", star_vertices())
    disk = InitialSet("disk", "S1 disk")

    star_samples = sample_initial_set(rng, star, N_SCHEMATIC_SAMPLES)
    disk_samples = sample_initial_set(rng, disk, N_SCHEMATIC_SAMPLES)
    star_boundary = polygon_boundary(star_vertices())
    circle_boundary = disk_boundary()

    panels = [
        ("initial", 0.0),
        ("t = 0.11", 0.11),
        ("t = 0.22", 0.22),
        ("t = 0.33", 0.33),
    ]

    star_sample_color = "#0077c8"
    disk_sample_color = "#d62828"
    star_boundary_color = "#023e8a"
    disk_boundary_color = "#9d0208"

    fig = plt.figure(figsize=(18.0, 16.0), constrained_layout=False)
    axes = np.array(
        [
            [
                fig.add_axes([0.10, 0.60, 0.34, 0.30]),
                fig.add_axes([0.48, 0.60, 0.34, 0.30]),
            ],
            [
                fig.add_axes([0.10, 0.24, 0.34, 0.30]),
                fig.add_axes([0.48, 0.24, 0.34, 0.30]),
            ],
        ]
    )
    legend_ax = fig.add_axes([0.12, 0.03, 0.76, 0.12])
    legend_ax.axis("off")
    for panel_index, (ax, (title, t)) in enumerate(zip(axes.ravel(), panels)):
        if t == 0.0:
            star_points_t = star_samples
            disk_points_t = disk_samples
            star_boundary_t = star_boundary
            disk_boundary_t = circle_boundary
        else:
            star_points_t = flow(star_samples, t)
            disk_points_t = flow(disk_samples, t)
            star_boundary_t = flow(star_boundary, t)
            disk_boundary_t = flow(circle_boundary, t)

        ax.scatter(
            disk_points_t[:, 0],
            disk_points_t[:, 1],
            s=56,
            alpha=0.88,
            color=disk_sample_color,
            linewidths=0,
            label="disk samples",
        )
        ax.scatter(
            star_points_t[:, 0],
            star_points_t[:, 1],
            s=56,
            alpha=0.90,
            color=star_sample_color,
            linewidths=0,
            label="star samples",
        )
        ax.plot(
            disk_boundary_t[:, 0],
            disk_boundary_t[:, 1],
            color=disk_boundary_color,
            linewidth=4.0,
            label="disk boundary",
        )
        ax.plot(
            star_boundary_t[:, 0],
            star_boundary_t[:, 1],
            color=star_boundary_color,
            linewidth=4.0,
            label="star boundary",
        )

        panel_points = np.vstack((disk_points_t, star_points_t, disk_boundary_t, star_boundary_t))
        lower = panel_points.min(axis=0)
        upper = panel_points.max(axis=0)
        span = upper - lower
        padding = np.maximum(0.08 * span, np.array([0.08, 0.08]))
        ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])

        ax.set_title(title, **SCHEMATIC_TITLE_FONT)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if panel_index < 2:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="both", labelsize=SCHEMATIC_TICK_SIZE)
        ax.set_box_aspect(1.0)
        ax.grid(True, alpha=0.25)

    legend_ax.scatter(
        [0.03],
        [0.66],
        s=260,
        color=disk_sample_color,
        alpha=0.88,
        transform=legend_ax.transAxes,
        clip_on=False,
    )
    legend_ax.text(
        0.09,
        0.68,
        "disk samples",
        va="center",
        ha="left",
        transform=legend_ax.transAxes,
        **SCHEMATIC_LABEL_FONT,
    )
    legend_ax.plot(
        [0.41, 0.53],
        [0.68, 0.68],
        color=disk_boundary_color,
        linewidth=4.0,
        transform=legend_ax.transAxes,
        clip_on=False,
    )
    legend_ax.text(
        0.57,
        0.68,
        "disk boundary",
        va="center",
        ha="left",
        transform=legend_ax.transAxes,
        **SCHEMATIC_LABEL_FONT,
    )
    legend_ax.scatter(
        [0.03],
        [0.25],
        s=260,
        color=star_sample_color,
        alpha=0.90,
        transform=legend_ax.transAxes,
        clip_on=False,
    )
    legend_ax.text(
        0.09,
        0.22,
        "star samples",
        va="center",
        ha="left",
        transform=legend_ax.transAxes,
        **SCHEMATIC_LABEL_FONT,
    )
    legend_ax.plot(
        [0.41, 0.53],
        [0.22, 0.22],
        color=star_boundary_color,
        linewidth=4.0,
        transform=legend_ax.transAxes,
        clip_on=False,
    )
    legend_ax.text(
        0.57,
        0.22,
        "star boundary",
        va="center",
        ha="left",
        transform=legend_ax.transAxes,
        **SCHEMATIC_LABEL_FONT,
    )
    fig.savefig(SAMPLE_FLOW_FIGURE_PATH, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    initial_sets = (
        InitialSet("disk", "S1 disk"),
        InitialSet("star", "S2 star", star_vertices()),
    )

    true_clouds: dict[tuple[str, float], np.ndarray] = {}
    for initial_set in initial_sets:
        dense_initial = sample_initial_set(rng, initial_set, N_TRUE_POINTS)
        for t in TIMES:
            true_clouds[(initial_set.name, t)] = flow(dense_initial, t)

    results: dict[tuple[str, float, int], float] = {}
    for initial_set in initial_sets:
        for t in TIMES:
            true_points_t = true_clouds[(initial_set.name, t)]
            for n in SAMPLE_SIZES:
                distances = np.empty(N_TRIALS, dtype=float)
                for trial in range(N_TRIALS):
                    distances[trial] = run_trial(rng, initial_set, t, n, true_points_t)
                results[(initial_set.name, t, n)] = float(np.mean(distances))
                print(
                    f"finished {initial_set.label}, t={t:.2f}, N={n}: "
                    f"mean d_H={results[(initial_set.name, t, n)]:.6f}"
                )

    print()
    print_table(results)
    plot_results(results)
    plot_sample_flow_schematic()
    print(f"\nsaved figure to {FIGURE_PATH}")
    print(f"saved sample flow schematic to {SAMPLE_FLOW_FIGURE_PATH}")


if __name__ == "__main__":
    main()
