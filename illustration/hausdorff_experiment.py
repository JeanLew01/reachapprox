"""Reachable-set approximation experiment for dx/dt = x^2, dy/dt = 0.

Run from the repository root with:

    python reachapprox/illustration/hausdorff_experiment.py

The script prints the mean approximate Hausdorff distances and saves
``hausdorff_vs_samples.png`` in the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from reachapprox.exp.quaddynadv.fun import flow
from reachapprox.exp.quaddynadv.fun.support_estimators import (
    approximate_support_hausdorff as approximate_hausdorff,
    christoffel_estimator_mask,
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix"
    }
)

TITLE_SIZE = 24
LABEL_SIZE = 24
TICK_SIZE = 15
LEGEND_SIZE = 17

PLOT_FONT = {"fontname": "DejaVu Serif"}
PLOT_FONT_PROP = {"family": "DejaVu Serif", "size": LEGEND_SIZE}

CENTER = np.array([2.0, 0.0])
OUTER_RADIUS = 1.0
INNER_RADIUS = OUTER_RADIUS * np.sin(np.pi / 10.0) / np.sin(3.0 * np.pi / 10.0)

TIMES = tuple(float(t) for t in np.linspace(0.01, 0.33, 17))
SAMPLE_SIZES = (100, 1000)
N_TRIALS = 50

GRID_RESOLUTION = 170
N_TRUE_POINTS = 35_000
N_SCHEMATIC_SAMPLES = 500
RANDOM_SEED = 7

RESULTS_DIR = Path("reachapprox/illustration/results")
FIGURE_PATH = RESULTS_DIR / "hausdorff_vs_samples.png"
SAMPLE_FLOW_FIGURE_PATH = RESULTS_DIR / "sample_flow_schematic.png"
SUPPORT_FIGURE_TEMPLATE = str(RESULTS_DIR / "christoffel_support_N{n_samples}.png")

@dataclass(frozen=True)
class InitialSet:
    name: str
    label: str
    polygon: np.ndarray | None = None


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


def convex_hull_estimator_mask(samples: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    """Classify grid points inside the convex hull of the endpoint samples."""
    hull = ConvexHull(samples)
    hull_vertices = samples[hull.vertices]
    path = MplPath(hull_vertices)
    mask = path.contains_points(grid_points, radius=1e-12)

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
    mask = convex_hull_estimator_mask(endpoint_samples, grid_points)
    estimated_points = grid_points[mask]
    return approximate_hausdorff(true_points_t, estimated_points)


def print_table(results: dict[tuple[str, float, int], float]) -> None:
    header = f"{'set':<8} {'t':>8} {'N':>6} {'mean Hausdorff':>18}"
    print(header)
    print("-" * len(header))
    for set_name in ("disk", "star"):
        for t in TIMES:
            for n in SAMPLE_SIZES:
                value = results[(set_name, t, n)]
                print(f"{set_name:<8} {t:>8.4f} {n:>6d} {value:>18.6f}")


def confidence_bounds(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean and 95% confidence interval bounds."""
    mean = float(np.mean(values))
    stderr = float(np.std(values, ddof=1) / np.sqrt(values.size))
    half_width = 1.96 * stderr
    lower = max(mean - half_width, np.finfo(float).tiny)
    upper = mean + half_width
    return mean, lower, upper


def plot_results(
    results: dict[tuple[str, float, int], float],
    ci_bounds: dict[tuple[str, float, int], tuple[float, float]] | None = None,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)

    set_colors = {
        "disk": "#d62728",
        "star": "#1f77b4",
    }
    sample_linestyles = {
        100: "-",
        1000: "--",
    }
    markers = {
        100: "s",
        1000: "D",
    }

    for n in SAMPLE_SIZES:
        for set_name, set_label in (
            ("disk", "S1 disk"),
            ("star", "S2 star"),
        ):
            ys = [results[(set_name, t, n)] for t in TIMES]
            color = set_colors[set_name]
            ax.plot(
                TIMES,
                ys,
                color=color,
                linestyle=sample_linestyles[n],
                marker=markers[n],
                linewidth=2.0,
                markersize=5.0,
                label=f"{set_label}, N = {n}",
            )
            if ci_bounds is not None:
                lowers = [ci_bounds[(set_name, t, n)][0] for t in TIMES]
                uppers = [ci_bounds[(set_name, t, n)][1] for t in TIMES]
                ax.fill_between(TIMES, lowers, uppers, color=color, alpha=0.22)

    ax.set_yscale("log")
    ax.set_xticks(TIMES)
    ax.set_xticklabels([f"{t:.4f}" for t in TIMES], rotation=35, ha="right", fontsize=TICK_SIZE, **PLOT_FONT)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_SIZE * 0.8)
    ax.set_xlabel("time t", fontsize=LABEL_SIZE, **PLOT_FONT)
    ax.set_ylabel("Hausdorff distance", fontsize=LABEL_SIZE, **PLOT_FONT)
    ax.set_title("Reachable-set approximation error", fontsize=TITLE_SIZE, **PLOT_FONT)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, prop=PLOT_FONT_PROP)
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_sample_flow_schematic() -> None:
    """Save a schematic of disk/star samples and exact transported boundaries."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

    star_sample_color = "#8ecae6"
    disk_sample_color = "#f4a6a6"
    star_boundary_color = "#023e8a"
    disk_boundary_color = "#9d0208"

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.8), constrained_layout=True)
    for ax, (title, t) in zip(axes.ravel(), panels):
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
            s=8,
            alpha=0.48,
            color=disk_sample_color,
            linewidths=0,
            label="disk samples",
        )
        ax.scatter(
            star_points_t[:, 0],
            star_points_t[:, 1],
            s=8,
            alpha=0.56,
            color=star_sample_color,
            linewidths=0,
            label="star samples",
        )
        ax.plot(
            disk_boundary_t[:, 0],
            disk_boundary_t[:, 1],
            color=disk_boundary_color,
            linewidth=2.2,
            label="disk boundary",
        )
        ax.plot(
            star_boundary_t[:, 0],
            star_boundary_t[:, 1],
            color=star_boundary_color,
            linewidth=2.2,
            label="star boundary",
        )

        panel_points = np.vstack((disk_points_t, star_points_t, disk_boundary_t, star_boundary_t))
        lower = panel_points.min(axis=0)
        upper = panel_points.max(axis=0)
        span = upper - lower
        padding = np.maximum(0.08 * span, np.array([0.08, 0.08]))
        ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])

        ax.set_title(title, **PLOT_FONT)
        ax.set_xlabel("x", **PLOT_FONT)
        ax.set_ylabel("y", **PLOT_FONT)
        ax.set_box_aspect(1.0)
        ax.grid(True, alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        prop=PLOT_FONT_PROP,
    )
    fig.savefig(SAMPLE_FLOW_FIGURE_PATH, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def plot_christoffel_support(n_samples: int) -> Path:
    """Save Christoffel support-set estimates over all time points."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    initial_sets = (
        InitialSet("disk", "S1 disk"),
        InitialSet("star", "S2 star", star_vertices()),
    )
    support_colors = {
        "disk": "#f4a6a6",
        "star": "#8ecae6",
    }
    boundary_colors = {
        "disk": "#9d0208",
        "star": "#023e8a",
    }

    initial_samples = {
        initial_set.name: sample_initial_set(rng, initial_set, n_samples)
        for initial_set in initial_sets
    }
    dense_initials = {
        initial_set.name: sample_initial_set(rng, initial_set, N_TRUE_POINTS)
        for initial_set in initial_sets
    }
    initial_boundaries = {
        "disk": disk_boundary(),
        "star": polygon_boundary(star_vertices()),
    }

    fig, axes = plt.subplots(
        2,
        len(TIMES),
        figsize=(26.0, 4.8),
        constrained_layout=True,
        squeeze=False,
    )

    for row, initial_set in enumerate(initial_sets):
        for col, t in enumerate(TIMES):
            ax = axes[row, col]
            samples_t = flow(initial_samples[initial_set.name], t)
            true_points_t = flow(dense_initials[initial_set.name], t)
            boundary_t = flow(initial_boundaries[initial_set.name], t)
            grid_points = make_grid(true_points_t, samples_t, GRID_RESOLUTION)
            mask = christoffel_estimator_mask(samples_t, grid_points)

            xs = grid_points[:GRID_RESOLUTION, 0]
            ys = grid_points[::GRID_RESOLUTION, 1]
            mask_grid = mask.reshape(GRID_RESOLUTION, GRID_RESOLUTION)
            ax.contourf(
                xs,
                ys,
                mask_grid.astype(float),
                levels=[0.5, 1.5],
                colors=[support_colors[initial_set.name]],
                alpha=0.62,
            )
            ax.plot(
                boundary_t[:, 0],
                boundary_t[:, 1],
                color=boundary_colors[initial_set.name],
                linewidth=1.5,
            )
            scatter_size = 14 if n_samples <= 10 else 6 if n_samples <= 100 else 2
            ax.scatter(samples_t[:, 0], samples_t[:, 1], s=scatter_size, color="black", linewidths=0)

            panel_points = np.vstack((grid_points[mask], boundary_t, samples_t))
            lower = panel_points.min(axis=0)
            upper = panel_points.max(axis=0)
            span = upper - lower
            padding = np.maximum(0.08 * span, np.array([0.04, 0.04]))
            ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            ax.set_box_aspect(1.0)
            ax.tick_params(axis="both", which="major", labelsize=8)

            if row == 0:
                ax.set_title(f"t={t:.4f}", fontsize=10, **PLOT_FONT)
            if col == 0:
                ax.set_ylabel(initial_set.label, fontsize=14, **PLOT_FONT)

    figure_path = Path(SUPPORT_FIGURE_TEMPLATE.format(n_samples=n_samples))
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return figure_path


def plot_christoffel_support_n10() -> Path:
    """Save Christoffel support-set estimates for N=10 over all time points."""
    return plot_christoffel_support(10)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    trial_seeds = RANDOM_SEED + np.arange(N_TRIALS)
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
    ci_bounds: dict[tuple[str, float, int], tuple[float, float]] = {}
    for initial_set in initial_sets:
        for t in TIMES:
            true_points_t = true_clouds[(initial_set.name, t)]
            for n in SAMPLE_SIZES:
                distances = np.empty(N_TRIALS, dtype=float)
                for trial, seed in enumerate(trial_seeds):
                    trial_rng = np.random.default_rng(int(seed))
                    distances[trial] = run_trial(trial_rng, initial_set, t, n, true_points_t)
                mean, lower, upper = confidence_bounds(distances)
                results[(initial_set.name, t, n)] = mean
                ci_bounds[(initial_set.name, t, n)] = (lower, upper)
                print(
                    f"finished {initial_set.label}, t={t:.4f}, N={n}: "
                    f"mean d_H={mean:.6f}, 95% CI=({lower:.6f}, {upper:.6f})"
                )

    print()
    print_table(results)
    plot_results(results, ci_bounds)
    plot_sample_flow_schematic()
    print(f"\nsaved figure to {FIGURE_PATH}")
    print(f"saved sample flow schematic to {SAMPLE_FLOW_FIGURE_PATH}")


if __name__ == "__main__":
    main()
