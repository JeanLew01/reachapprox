"""Figure 1 (Section 2): reachable-set approximation under x' = x^2, y' = 0.

A disk and a five-pointed star of the same circumradius, centered at (2, 0),
are sampled i.i.d. uniformly and propagated by the exact (non-globally-
Lipschitz) flow phi_T(x, y) = (x / (1 - T x), y).  The estimator is the convex
hull of the endpoints, discretized on a 170 x 170 grid, and the error is the
symmetric Hausdorff distance to a dense cloud of 35,000 propagated points,
averaged over 50 trials.

Outputs (results/fig1/): snapshots, error vs. sample size (T = 0.22), and
error vs. time (N = 1000), saved as three panels.

    python -m experiments.fig1_quadratic_illustration
    python -m experiments.fig1_quadratic_illustration --plot-only
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from reachapprox.estimators import convex_hull_mask, make_grid
from reachapprox.flows import quadratic_flow
from reachapprox.geometry import CENTER, DISK_RADIUS, boundary_points, sample_disk, sample_uniform_polygon, star_set
from reachapprox.metrics import cloud_hausdorff
from reachapprox.utils import RESULTS_DIR, mean_ci95, power_of_ten_labels, read_csv, use_serif_fonts, write_csv


TIMES = tuple(float(t) for t in np.linspace(0.01, 0.33, 17))
TIME_SWEEP_SAMPLE_SIZE = 1_000
SAMPLE_SIZES = (10, 30, 100, 300, 1_000, 3_000, 10_000)
SAMPLE_SWEEP_TIME = 0.22
SNAPSHOT_TIMES = (0.0, 0.11, 0.22, 0.33)
N_TRIALS = 50
GRID_RESOLUTION = 170
N_TRUE_POINTS = 35_000
N_SNAPSHOT_SAMPLES = 500
RANDOM_SEED = 7

OUT_DIR = RESULTS_DIR / "fig1"
TRIALS_CSV = OUT_DIR / "fig1_trials.csv"
SNAPSHOT_FIGURE = OUT_DIR / "fig1_snapshots.png"
SAMPLES_FIGURE = OUT_DIR / "fig1_error_vs_samples.png"
TIME_FIGURE = OUT_DIR / "fig1_error_vs_time.png"

STYLES = {
    "disk": {"label": "disk", "color": "#d62728", "marker": "o"},
    "star": {"label": "star", "color": "#005a9c", "marker": "s"},
}
STAR = star_set(DISK_RADIUS)


def sample_initial_set(rng: np.random.Generator, name: str, n: int) -> np.ndarray:
    if name == "disk":
        return sample_disk(rng, n)
    return sample_uniform_polygon(rng, STAR, n, batch_size=max(4 * n, 1_000))


def initial_boundary(name: str) -> np.ndarray:
    if name == "disk":
        angles = np.linspace(0.0, 2.0 * np.pi, 1_000, endpoint=False)
        return CENTER + DISK_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))
    return boundary_points(STAR, 1_200)


def run_trial(rng: np.random.Generator, name: str, T: float, n_samples: int, true_points: np.ndarray) -> float:
    endpoints = quadratic_flow(sample_initial_set(rng, name, n_samples), T)
    grid_points = make_grid(true_points, endpoints, GRID_RESOLUTION)
    return cloud_hausdorff(true_points, grid_points[convex_hull_mask(endpoints, grid_points)])


def run_experiment() -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    dense_initial = {name: sample_initial_set(rng, name, N_TRUE_POINTS) for name in STYLES}
    rows = []
    for name in STYLES:
        conditions = [("time", T, TIME_SWEEP_SAMPLE_SIZE) for T in TIMES]
        conditions += [("sample", SAMPLE_SWEEP_TIME, N) for N in SAMPLE_SIZES]
        for sweep, T, N in conditions:
            true_points = quadratic_flow(dense_initial[name], T)
            errors = []
            for trial in range(N_TRIALS):
                seed = RANDOM_SEED + trial
                errors.append(run_trial(np.random.default_rng(seed), name, T, N, true_points))
                rows.append({"sweep": sweep, "set": name, "T": T, "N": N, "trial": trial, "seed": seed,
                             "error": errors[-1]})
            print(f"{sweep:<6} {name:<4} T={T:.4f} N={N}: mean symmetric d_H={np.mean(errors):.6f}")
    return rows


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        coordinate = float(row["T"]) if row["sweep"] == "time" else int(float(row["N"]))
        grouped.setdefault((row["sweep"], row["set"], coordinate), []).append(float(row["error"]))
    return {key: tuple(float(v) for v in mean_ci95(np.array(errors))) for key, errors in grouped.items()}


def plot_snapshots() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    samples = {name: sample_initial_set(rng, name, N_SNAPSHOT_SAMPLES) for name in STYLES}
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.2), gridspec_kw={"hspace": 0.25, "wspace": 0.26})
    for index, (ax, T) in enumerate(zip(axes.ravel(), SNAPSHOT_TIMES)):
        plotted = []
        for name, style in STYLES.items():
            points = samples[name] if T == 0.0 else quadratic_flow(samples[name], T)
            boundary = initial_boundary(name) if T == 0.0 else quadratic_flow(initial_boundary(name), T)
            plotted += [points, boundary]
            ax.scatter(points[:, 0], points[:, 1], s=9, alpha=0.58, color=style["color"], linewidths=0,
                       label=f"{style['label']} samples")
            ax.plot(boundary[:, 0], boundary[:, 1], color=style["color"], linewidth=2.3,
                    label=f"{style['label']} boundary")
        panel = np.vstack(plotted)
        lower, upper = panel.min(axis=0), panel.max(axis=0)
        padding = np.maximum(0.07 * (upper - lower), 0.06)
        ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        ax.set_title("initial" if T == 0.0 else rf"$T={T:.2f}$", fontsize=25)
        if index >= 2:
            ax.set_xlabel("x", fontsize=23)
        if index % 2 == 0:
            ax.set_ylabel("y", fontsize=23)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_box_aspect(1.0)
        ax.grid(True, alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc="upper left", bbox_to_anchor=(-1.576, -0.170), ncol=2,
                      frameon=False, fontsize=20, handlelength=2.3, columnspacing=1.0,
                      markerscale=np.sqrt(8.0))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.22)
    fig.savefig(SNAPSHOT_FIGURE, dpi=220)
    plt.close(fig)


def plot_error(stats: dict, sweep: str, xs: tuple, title: str, xlabel: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.2), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.12)
    for name, style in STYLES.items():
        mean, lower, upper = (np.array([stats[(sweep, name, x)][k] for x in xs]) for k in range(3))
        ax.plot(xs, mean, color=style["color"], marker=style["marker"], linewidth=2.6, markersize=6.5,
                label=style["label"])
        ax.fill_between(xs, lower, upper, color=style["color"], alpha=0.20)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel("Hausdorff distance", fontsize=30)
    ax.set_title(title, fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.grid(True, which="both", alpha=0.26)
    ax.legend(frameon=False, fontsize=22)
    ax.set_box_aspect(1.0)
    if sweep == "time":
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{t:.3f}" for t in xs], rotation=45, ha="right")
    else:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(power_of_ten_labels(xs), rotation=40, ha="right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="re-plot from the saved trials CSV")
    args = parser.parse_args()
    use_serif_fonts("DejaVu Serif")

    if args.plot_only:
        rows = read_csv(TRIALS_CSV)
    else:
        rows = run_experiment()
        write_csv(rows, TRIALS_CSV)
        print(f"saved {TRIALS_CSV}")
    stats = aggregate(rows)
    plot_snapshots()
    plot_error(stats, "sample", SAMPLE_SIZES, rf"Error vs. samples ($T={SAMPLE_SWEEP_TIME:.2f}$)",
               r"sample size $N$", SAMPLES_FIGURE)
    plot_error(stats, "time", TIMES, rf"Error vs. time ($N={TIME_SWEEP_SAMPLE_SIZE}$)", r"time $T$", TIME_FIGURE)
    for path in (SNAPSHOT_FIGURE, SAMPLES_FIGURE, TIME_FIGURE):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
