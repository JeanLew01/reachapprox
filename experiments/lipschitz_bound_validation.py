"""Figure 2 (Section 5.1): quantitative validation of the bounds for x' = x, y' = 0.

Three equal-area initial sets (circle, opened triangle, triangle) centered at
(2, 0) are sampled i.i.d. uniformly, propagated by phi_T(x, y) = (e^T x, y), and
estimated by the convex hull of the endpoints.  The error is the symmetric
Hausdorff distance between the hull boundary and a dense reference boundary,
averaged over 50 trials.  The theoretical curves are the upper bound of
Theorem 1 and the lower bound of Theorem 2 solved for r (n=2, L=1, R=1,
rho=1, delta=0.05, r0=min(r0_circle, r0_opened)).

    python -m experiments.lipschitz_bound_validation              # run + plot
    python -m experiments.lipschitz_bound_validation --plot-only  # plot from CSV
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.special import lambertw

from reachapprox.estimators import convex_hull_polygon
from reachapprox.flows import linear_flow
from reachapprox.geometry import (
    DISK_RADIUS,
    boundary_points,
    equal_area_initial_sets,
    opened_triangle_reach,
    sample_uniform_polygon,
)
from reachapprox.metrics import boundary_hausdorff
from reachapprox.utils import (
    RESULTS_DIR,
    mean_ci95,
    parallel_map,
    power_of_ten_labels,
    read_csv,
    stable_seed_offset,
    use_serif_fonts,
    write_csv,
)


N_DIM = 2
LIPSCHITZ_L = 1.0
RHO = 1.0
DELTA = 0.05
COMMON_R = 1.0
THEOREM_R0 = min(DISK_RADIUS, opened_triangle_reach())

TIME_SAMPLE_SIZE = 1_000
TIME_GRID = tuple(float(t) for t in np.linspace(0.01, 2.0, 17))
FIXED_TIME = 1.0
SAMPLE_SIZES = (300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000)
N_TRIALS = 50
RANDOM_SEED = 11
REFERENCE_BOUNDARY_POINTS = 3_000
ESTIMATE_BOUNDARY_POINTS = 1_500

SNAPSHOT_TIMES = (0.0, 0.5, 1.0, 2.0)
SNAPSHOT_SAMPLES = 300
SNAPSHOT_LANES = {"disk": 10.0 / 3.0, "opened": 0.0, "triangle": -10.0 / 3.0}
SNAPSHOT_LABEL_Y = {"disk": 1.95, "opened": -1.30, "triangle": -4.60}

OUT_DIR = RESULTS_DIR / "lipschitz_bound_validation"
TRIALS_CSV = OUT_DIR / "trials.csv"
FIGURE_PATH = OUT_DIR / "lipschitz_bound_validation.png"

COLORS = {"disk": "#d62728", "opened": "#0072b2", "triangle": "#e69f00"}
MARKERS = {"disk": ("o", COLORS["disk"]), "opened": ("^", "white"), "triangle": ("^", COLORS["triangle"])}


def trial_seed(geometry: str, sweep: str, coordinate: float, trial: int) -> int:
    return RANDOM_SEED + stable_seed_offset(geometry, f"figure2-{sweep}-uniform", float(coordinate), 0) + trial


def run_condition(task: tuple[str, str, float, int]) -> list[dict]:
    """All trials of one (initial set, sweep, T, N) condition."""
    set_name, sweep, time, sample_size = task
    coordinate = time if sweep == "time" else sample_size
    geom = next(item.geom for item in equal_area_initial_sets() if item.name == set_name)
    reference = linear_flow(boundary_points(geom, REFERENCE_BOUNDARY_POINTS), time)
    rows = []
    for trial in range(N_TRIALS):
        seed = trial_seed(set_name, sweep, coordinate, trial)
        endpoints = linear_flow(sample_uniform_polygon(np.random.default_rng(seed), geom, sample_size), time)
        error = boundary_hausdorff(reference, convex_hull_polygon(endpoints), ESTIMATE_BOUNDARY_POINTS)
        rows.append({"sweep": sweep, "geometry": set_name, "T": time, "N": sample_size, "trial": trial,
                     "seed": seed, "error": error})
    return rows


def run_experiment(initial_sets, workers: int | None) -> list[dict]:
    tasks = []
    for item in initial_sets:
        tasks += [(item.name, "time", T, TIME_SAMPLE_SIZE) for T in TIME_GRID]
        tasks += [(item.name, "sample", FIXED_TIME, N) for N in SAMPLE_SIZES]
    rows = []
    for (set_name, sweep, time, sample_size), task_rows in zip(tasks, parallel_map(run_condition, tasks, workers)):
        mean = np.mean([row["error"] for row in task_rows])
        print(f"{sweep:<6} {set_name:<8} T={time:.4f} N={sample_size}: mean={mean:.6f}")
        rows += task_rows
    return rows


def aggregate(rows: list[dict]) -> dict:
    """(sweep, geometry, coordinate) -> (mean, ci_low, ci_high)."""
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        sweep = row["sweep"]
        coordinate = float(row["T"]) if sweep == "time" else int(float(row["N"]))
        grouped.setdefault((sweep, row["geometry"], coordinate), []).append(float(row["error"]))
    return {key: tuple(float(v) for v in mean_ci95(np.array(errors))) for key, errors in grouped.items()}


def upper_scale(sample_size, time) -> np.ndarray:
    """Theorem 1 solved for r: N = (2^{2n} e^{nLT} R^n / (rho r^n)) log(2^{3n} e^{nLT} R^n / (r^n delta))."""
    sample_size = np.asarray(sample_size, dtype=float)
    time = np.asarray(time, dtype=float)
    inside = (2.0**N_DIM) * RHO * sample_size / DELTA
    numerator = (2.0 ** (2 * N_DIM)) * np.exp(N_DIM * LIPSCHITZ_L * time) * COMMON_R**N_DIM
    return (numerator * np.real(lambertw(inside)) / (RHO * sample_size)) ** (1.0 / N_DIM)


def lower_scale(sample_size, time) -> np.ndarray:
    """Theorem 2 solved for r, masked outside its admissible range r <= 2^{-(n+1)/n} e^{LT} r0."""
    sample_size = np.asarray(sample_size, dtype=float)
    time = np.asarray(time, dtype=float)
    numerator = np.exp(N_DIM * LIPSCHITZ_L * time) * COMMON_R**N_DIM
    values = (numerator * np.log(1.0 / (2.0 * DELTA)) / (2.0 ** (N_DIM + 1) * sample_size)) ** (1.0 / N_DIM)
    limit = 2.0 ** (-(N_DIM + 1) / N_DIM) * np.exp(LIPSCHITZ_L * time) * THEOREM_R0
    return np.where(values <= limit, values, np.nan)


def plot_theory(ax, xs, upper, theory_time, lower) -> None:
    """Upper bound in black where Theorem 1 applies (r <= 2 e^{LT} r0), grey elsewhere."""
    xs = np.asarray(xs, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = upper <= 2.0 * np.exp(LIPSCHITZ_L * np.asarray(theory_time, dtype=float)) * THEOREM_R0
    ax.plot(xs, upper, color="#aaaaaa", linestyle="-.", linewidth=2.5, label="_nolegend_")
    valid_upper = np.where(valid, upper, np.nan)
    if np.any(valid):
        first_valid = int(np.flatnonzero(valid)[0])
        if first_valid > 0:
            valid_upper[first_valid - 1] = upper[first_valid - 1]
    ax.plot(xs, valid_upper, color="black", linestyle="-.", linewidth=2.5, label="Upper bound")
    ax.plot(xs, lower, color="#555555", linestyle="--", linewidth=2.3, label="Lower bound")


def plot_empirical(ax, xs, stats, sweep: str, initial_sets) -> None:
    for item in initial_sets:
        means, lowers, uppers = (np.array([stats[(sweep, item.name, x)][k] for x in xs]) for k in range(3))
        marker, face = MARKERS[item.name]
        ax.plot(xs, means, color=COLORS[item.name], marker=marker, markerfacecolor=face,
                markeredgecolor=COLORS[item.name], markeredgewidth=1.8, markersize=7.5, linewidth=2.2,
                label=item.label)
        ax.fill_between(xs, lowers, uppers, color=COLORS[item.name], alpha=0.16)


def draw_snapshot(ax, time: float, initial_sets, samples, boundaries, xlim_final, row: int, col: int) -> None:
    """One snapshot; each geometry is drawn in its own display-only vertical lane."""
    for item in initial_sets:
        marker, face = MARKERS[item.name]
        points = linear_flow(samples[item.name], time) + np.array([0.0, SNAPSHOT_LANES[item.name]])
        boundary = linear_flow(boundaries[item.name], time) + np.array([0.0, SNAPSHOT_LANES[item.name]])
        ax.scatter(points[:, 0], points[:, 1], s=10, marker=marker, facecolors=face,
                   edgecolors=COLORS[item.name], linewidths=0.7, alpha=0.55)
        ax.plot(boundary[:, 0], boundary[:, 1], color=COLORS[item.name], linewidth=1.8)
        ax.text(0.025, SNAPSHOT_LABEL_Y[item.name], item.label, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=15, color=COLORS[item.name],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0})
    for lane_boundary in (-5.0 / 3.0, 5.0 / 3.0):
        ax.axhline(lane_boundary, color="#777777", linestyle="--", linewidth=1.0, alpha=0.65)

    ax.set_xlim(*(xlim_final if time == SNAPSHOT_TIMES[-1] else (0.0, 10.0)))
    ax.set_ylim(-5.0, 5.0)
    ax.set_title("initial" if time == 0.0 else rf"$T={time:g}$", fontsize=25)
    if row == 1:
        ax.set_xlabel("x", fontsize=23)
    if col == 0:
        ax.set_ylabel("display y", fontsize=23)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", labelsize=19)
    ax.grid(True, alpha=0.22)
    ax.set_box_aspect(1.0)


def plot_figure(stats, initial_sets) -> None:
    fig = plt.figure(figsize=(23.5, 7.4))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.30, 1.15, 1.15), left=0.025, right=0.995,
                            top=0.955, bottom=0.12, wspace=0.18)

    samples, boundaries = {}, {}
    for item in initial_sets:
        rng = np.random.default_rng(
            RANDOM_SEED + stable_seed_offset(item.name, "figure2-snapshot-uniform", 0.0, SNAPSHOT_SAMPLES)
        )
        samples[item.name] = sample_uniform_polygon(rng, item.geom, SNAPSHOT_SAMPLES)
        boundaries[item.name] = boundary_points(item.geom, 1_200)
    final = np.vstack([linear_flow(b, t) for t in SNAPSHOT_TIMES for b in boundaries.values()])
    pad = np.maximum(0.07 * (final.max(axis=0) - final.min(axis=0)), 0.05)
    xlim_final = (final[:, 0].min() - pad[0], final[:, 0].max() + pad[0])

    snapshot_grid = grid[0].subgridspec(2, 2, hspace=0.25, wspace=0.12)
    for index, time in enumerate(SNAPSHOT_TIMES):
        row, col = divmod(index, 2)
        draw_snapshot(fig.add_subplot(snapshot_grid[row, col]), time, initial_sets, samples, boundaries,
                      xlim_final, row, col)

    ax_samples, ax_time = fig.add_subplot(grid[1]), fig.add_subplot(grid[2])
    plot_empirical(ax_samples, SAMPLE_SIZES, stats, "sample", initial_sets)
    plot_theory(ax_samples, SAMPLE_SIZES, upper_scale(np.asarray(SAMPLE_SIZES), FIXED_TIME), FIXED_TIME,
                lower_scale(np.asarray(SAMPLE_SIZES), FIXED_TIME))
    ax_samples.set_xscale("log")
    ax_samples.set_xlabel(r"sample size $N$")
    ax_samples.set_title(rf"Error vs. samples ($T={FIXED_TIME:g}$)")
    ax_samples.set_xticks(SAMPLE_SIZES)
    ax_samples.set_xticklabels(power_of_ten_labels(SAMPLE_SIZES), rotation=35, ha="right")

    times = np.asarray(TIME_GRID)
    plot_empirical(ax_time, TIME_GRID, stats, "time", initial_sets)
    plot_theory(ax_time, TIME_GRID, upper_scale(TIME_SAMPLE_SIZE, times), times,
                lower_scale(TIME_SAMPLE_SIZE, times))
    ax_time.set_xlabel(r"time $T$")
    ax_time.set_title(rf"Error vs. time ($N={TIME_SAMPLE_SIZE}$)")
    ax_time.set_xticks(TIME_GRID[::2])
    ax_time.set_xticklabels([f"{t:.2f}" for t in TIME_GRID[::2]], rotation=35, ha="right")

    axes = (ax_samples, ax_time)
    y_low = min(float(ax.dataLim.intervaly[0]) for ax in axes)
    y_high = max(float(ax.dataLim.intervaly[1]) for ax in axes)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(0.75 * y_low, 1.25 * y_high)
        ax.set_ylabel("Hausdorff distance", fontsize=28)
        ax.xaxis.label.set_size(28)
        ax.title.set_size(28)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.tick_params(axis="both", which="minor", labelsize=16)
        ax.grid(True, which="both", alpha=0.25)
    ax_time.legend(loc="lower right", frameon=False, fontsize=22, borderpad=0.7, labelspacing=0.55,
                   handlelength=2.6)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="re-plot from the saved trials CSV")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    args = parser.parse_args()
    use_serif_fonts("STIXGeneral")
    initial_sets = equal_area_initial_sets()
    print(f"r0(circle)={DISK_RADIUS:.6f}, r0(opened triangle)={opened_triangle_reach():.6f}, r0(triangle)=0")

    if args.plot_only:
        rows = read_csv(TRIALS_CSV)
    else:
        rows = run_experiment(initial_sets, args.workers)
        write_csv(rows, TRIALS_CSV)
        print(f"saved {TRIALS_CSV}")
    plot_figure(aggregate(rows), initial_sets)
    print(f"saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
