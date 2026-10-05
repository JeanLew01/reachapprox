"""Dynamics: the horizon enters through the one-sided Lipschitz constant mu, not through L.

The equilateral triangle of area pi centered at (2, 0), a polytope without
positive reach, is sampled i.i.d. uniformly and propagated through four planar
fields with closed-form flows:

    expanding spiral   x' = [[ 0.5, -4], [4,  0.5]] x    mu =  0.5, L = 4.03
    rotation           x' = [[ 0,   -4], [4,  0  ]] x    mu =  0,   L = 4
    spiral sink        x' = [[-1,   -4], [4, -1  ]] x    mu = -1,   L = 4.12
    cubic damping      x' = -|x|^2 x                     mu =  0,   not globally Lipschitz

The estimator is the endpoint cloud and the error is its exact Hausdorff
distance to the reachable set, over 200 trials, for T in [0, 2] at N = 1000
and for N from 100 to 30000 at T = 1.  The theory curves are the upper bound
solved for r, with e^{mu T} or, for the cubic field, with the Lipschitz
constant (1 + 2 T rho_min^2)^{-1/2} of its flow map on the initial set
(rho_min is the distance from the origin to the triangle).

    python -m experiments.one_sided_lipschitz_horizon              # run + plot
    python -m experiments.one_sided_lipschitz_horizon --plot-only  # plot from CSV
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from reachapprox.bounds import certified_accuracy
from reachapprox.flows import cubic_damping_flow, spiral_flow
from reachapprox.geometry import (
    contains_points,
    convex_polygon_standardness,
    dense_boundary,
    equilateral_triangle,
    sample_uniform_polygon,
)
from reachapprox.metrics import cloud_inner_error
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
RHO = 1.0
DELTA = 0.05
COMMON_R = 1.0
OMEGA = 4.0

TRIANGLE = equilateral_triangle()
TRIANGLE_VERTICES = np.asarray(TRIANGLE.exterior.coords[:-1], dtype=float)
TRIANGLE_KAPPA, TRIANGLE_LAMBDA = convex_polygon_standardness(TRIANGLE)
TRIANGLE_DIAMETER = float(np.linalg.norm(TRIANGLE_VERTICES[0] - TRIANGLE_VERTICES[1]))
RHO_MIN = Point(0.0, 0.0).distance(TRIANGLE)
RHO_MAX = float(np.max(np.linalg.norm(TRIANGLE_VERTICES, axis=1)))

TIME_GRID = tuple(float(t) for t in np.linspace(0.0, 2.0, 13))
TIME_SAMPLE_SIZE = 1_000
FIXED_TIME = 1.0
SAMPLE_SIZES = (100, 300, 1_000, 3_000, 10_000, 30_000)
N_TRIALS = 200
RANDOM_SEED = 31
BOUNDARY_SPACING = 2e-4
SNAPSHOT_SAMPLES = 250

OUT_DIR = RESULTS_DIR / "one_sided_lipschitz_horizon"
TRIALS_CSV = OUT_DIR / "trials.csv"
RATES_CSV = OUT_DIR / "growth_rates.csv"
FIGURE_PATH = OUT_DIR / "one_sided_lipschitz_horizon.png"

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"


@dataclass(frozen=True)
class System:
    label: str
    color: str
    marker: str
    mu: float  # one-sided Lipschitz constant of the field
    lipschitz: float  # Lipschitz constant of the field on the ball that contains the trajectories
    flow: Callable[[np.ndarray, float], np.ndarray]
    preimage_in_range: Callable[[np.ndarray, float], np.ndarray]  # mask of the points in phi_T(R^2)
    flow_lipschitz: Callable[[float], float]  # certified Lipschitz constant of phi_T on the initial set


def spiral_system(label: str, color: str, marker: str, rate: float) -> System:
    return System(label, color, marker, rate, float(np.hypot(rate, OMEGA)),
                  partial(spiral_flow, rate=rate, omega=OMEGA),
                  lambda points, T: np.ones(points.shape[0], dtype=bool),
                  lambda T: float(np.exp(rate * T)))


SYSTEMS = {
    "expanding": spiral_system("expanding spiral", "#eb6834", "o", 0.5),
    "rotation": spiral_system("rotation", "#1baf7a", "s", 0.0),
    "sink": spiral_system("spiral sink", "#2a78d6", "^", -1.0),
    "cubic": System("cubic damping", "#4a3aa7", "D", 0.0, 3.0 * RHO_MAX**2, cubic_damping_flow,
                    lambda points, T: 2.0 * T * np.sum(points * points, axis=1) < 1.0,
                    lambda T: float((1.0 + 2.0 * T * RHO_MIN**2) ** -0.5)),
}


def reachable_set_contains(system: System, points: np.ndarray, T: float) -> np.ndarray:
    """Membership in S_T = phi_T(S_0), through the inverse flow."""
    inside = np.zeros(points.shape[0], dtype=bool)
    in_range = system.preimage_in_range(points, T)
    inside[in_range] = contains_points(TRIANGLE, system.flow(points[in_range], -T))
    return inside


def trial_seed(name: str, sweep: str, coordinate: float, trial: int) -> int:
    return RANDOM_SEED + stable_seed_offset(name, f"osl-{sweep}", float(coordinate), 0) + trial


def run_condition(task: tuple[str, str, float, int]) -> list[dict]:
    """All trials of one (system, sweep, T, N) condition."""
    name, sweep, T, sample_size = task
    system = SYSTEMS[name]
    coordinate = T if sweep == "time" else sample_size
    boundary = system.flow(dense_boundary(TRIANGLE, BOUNDARY_SPACING), T)
    rows = []
    for trial in range(N_TRIALS):
        seed = trial_seed(name, sweep, coordinate, trial)
        endpoints = system.flow(sample_uniform_polygon(np.random.default_rng(seed), TRIANGLE, sample_size), T)
        error = cloud_inner_error(endpoints, boundary, partial(reachable_set_contains, system, T=T))
        rows.append({"sweep": sweep, "system": name, "T": T, "N": sample_size, "trial": trial, "seed": seed,
                     "error": error})
    return rows


def run_experiment(workers: int | None) -> list[dict]:
    tasks = []
    for name in SYSTEMS:
        tasks += [(name, "time", T, TIME_SAMPLE_SIZE) for T in TIME_GRID]
        tasks += [(name, "sample", FIXED_TIME, n) for n in SAMPLE_SIZES]
    tasks.sort(key=lambda task: -task[3])  # longest conditions first
    rows = []
    for (name, sweep, T, n), task_rows in zip(tasks, parallel_map(run_condition, tasks, workers)):
        mean = np.mean([row["error"] for row in task_rows])
        print(f"{sweep:<6} {name:<9} T={T:.4f} N={n:>5}: mean d_H={mean:.5f}", flush=True)
        rows += task_rows
    return rows


def aggregate(rows: list[dict]) -> dict:
    """(sweep, system, coordinate) -> (mean, ci_low, ci_high, q95)."""
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        coordinate = round(float(row["T"]), 6) if row["sweep"] == "time" else int(float(row["N"]))
        grouped.setdefault((row["sweep"], row["system"], coordinate), []).append(float(row["error"]))
    return {key: (*(float(v) for v in mean_ci95(np.array(errors))), float(np.quantile(errors, 0.95)))
            for key, errors in grouped.items()}


def upper_bound(system: System, sample_size, T: float) -> np.ndarray:
    """Upper bound solved for r, with the certified Lipschitz constant of phi_T in place of e^{mu T}."""
    return certified_accuracy(sample_size, 1.0, n=N_DIM, mu=float(np.log(system.flow_lipschitz(T))), R=COMMON_R,
                              kappa=TRIANGLE_KAPPA, lam=TRIANGLE_LAMBDA, diameter=TRIANGLE_DIAMETER, convex=True,
                              rho=RHO, delta=DELTA)


def fit_rates(stats: dict) -> list[dict]:
    """Exponential rate of the mean error in T, change from T = 0 to 2, and log-log slope in N."""
    times = [round(t, 6) for t in TIME_GRID]
    fits = []
    for name, system in SYSTEMS.items():
        means = np.array([stats[("time", name, t)][0] for t in times])
        sample_means = [stats[("sample", name, n)][0] for n in SAMPLE_SIZES]
        fits.append({
            "system": name, "mu": system.mu, "lipschitz": system.lipschitz,
            "fitted_rate": np.polyfit(times, np.log(means), 1)[0],
            "error_ratio_T2_over_T0": means[-1] / means[0],
            "flow_lipschitz_T2": system.flow_lipschitz(TIME_GRID[-1]),
            "exp_mu_T2": np.exp(system.mu * TIME_GRID[-1]),
            "exp_L_T2": np.exp(system.lipschitz * TIME_GRID[-1]),
            "slope_in_N": np.polyfit(np.log(SAMPLE_SIZES), np.log(sample_means), 1)[0],
        })
    return fits


def draw_snapshot(ax, name: str, row: int, col: int) -> None:
    """The initial triangle, its reachable set at T = 1 with endpoint samples, and the path of the centroid."""
    system = SYSTEMS[name]
    samples = sample_uniform_polygon(np.random.default_rng(RANDOM_SEED), TRIANGLE, SNAPSHOT_SAMPLES)
    outline = dense_boundary(TRIANGLE, 5e-3)[:-3]
    outline = np.vstack((outline, outline[:1]))
    centroid = np.array([[TRIANGLE.centroid.x, TRIANGLE.centroid.y]])
    path = np.vstack([system.flow(centroid, t) for t in np.linspace(0.0, FIXED_TIME, 200)])
    image, points = system.flow(outline, FIXED_TIME), system.flow(samples, FIXED_TIME)

    ax.fill(outline[:, 0], outline[:, 1], facecolor=MUTED, alpha=0.10, edgecolor="none")
    ax.plot(outline[:, 0], outline[:, 1], color=MUTED, linewidth=1.8)
    ax.plot(path[:, 0], path[:, 1], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.scatter(points[:, 0], points[:, 1], s=6, color=system.color, alpha=0.55, linewidths=0)
    ax.plot(image[:, 0], image[:, 1], color=system.color, linewidth=2.2)
    ax.plot([0.0], [0.0], marker="+", color=INK, markersize=10, markeredgewidth=1.4)
    ax.text(2.0, 1.75, r"$S_0$", fontsize=18, color=INK, ha="center", va="bottom")
    constants = rf"$\mu={system.mu:g}$, " + ("not Lipschitz" if name == "cubic" else rf"$L={system.lipschitz:.2f}$")
    ax.set_title(f"{system.label}\n{constants}", fontsize=19, linespacing=1.15)
    ax.set_xlim(-5.9, 3.9)
    ax.set_ylim(-5.6, 2.9)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2])
    ax.set_yticks([-4, -2, 0, 2])
    ax.tick_params(axis="both", labelsize=15, labelbottom=row == 1, labelleft=col == 0)
    ax.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def plot_figure(stats: dict) -> None:
    fig = plt.figure(figsize=(23.5, 7.4))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.15, 1.15), wspace=0.22)
    snapshot_grid = grid[0].subgridspec(2, 2, hspace=0.36, wspace=0.06)
    for index, name in enumerate(SYSTEMS):
        row, col = divmod(index, 2)
        draw_snapshot(fig.add_subplot(snapshot_grid[row, col]), name, row, col)
    ax_time, ax_samples = fig.add_subplot(grid[1]), fig.add_subplot(grid[2])

    times = np.asarray(TIME_GRID)
    sizes = np.asarray(SAMPLE_SIZES, dtype=float)
    for name, system in SYSTEMS.items():
        for ax, sweep, xs, keys in ((ax_time, "time", times, [round(t, 6) for t in TIME_GRID]),
                                    (ax_samples, "sample", sizes, SAMPLE_SIZES)):
            mean, low, high, _ = (np.array([stats[(sweep, name, key)][k] for key in keys]) for k in range(4))
            ax.plot(xs, mean, color=system.color, marker=system.marker, markersize=10, markeredgecolor="white",
                    markeredgewidth=1.2, linewidth=2.2, label=system.label)
            ax.fill_between(xs, low, high, color=system.color, alpha=0.16, linewidth=0)
        dense_times = np.linspace(times[0], times[-1], 81)
        ax_time.plot(dense_times, [upper_bound(system, TIME_SAMPLE_SIZE, t) for t in dense_times],
                     color=system.color, linestyle="--", linewidth=1.8)
        ax_samples.plot(sizes, upper_bound(system, sizes, FIXED_TIME), color=system.color, linestyle="--",
                        linewidth=1.8)
    # The same bound with the Lipschitz constant L of the spiral sink in place of mu.
    sink = SYSTEMS["sink"]
    dense_times = np.linspace(0.0, 0.6, 31)
    ax_time.plot(dense_times, upper_bound(sink, TIME_SAMPLE_SIZE, 0.0) * np.exp(sink.lipschitz * dense_times),
                 color=MUTED, linestyle=":", linewidth=2.4)
    ax_time.text(0.36, 1.45, r"$e^{LT}$ scaling," "\n" r"spiral sink", fontsize=18, color=MUTED, ha="left", va="center",
                 linespacing=1.2)
    ax_time.plot([], [], color=INK, linestyle="--", linewidth=1.8, label="upper bound (same color)")
    ax_time.set_xlabel(r"horizon $T$")
    ax_time.set_title(rf"Error vs. horizon ($N={TIME_SAMPLE_SIZE}$)")
    ax_time.set_ylim(2.5e-3, 3.0)
    ax_time.legend(loc="lower left", frameon=False, fontsize=17, handlelength=2.2, labelspacing=0.35, ncol=2,
                   columnspacing=1.0)

    ax_samples.set_xscale("log")
    ax_samples.set_xticks(SAMPLE_SIZES)
    ax_samples.set_xticklabels(power_of_ten_labels(SAMPLE_SIZES), rotation=35, ha="right")
    ax_samples.minorticks_off()
    ax_samples.set_xlabel(r"sample size $N$")
    ax_samples.set_title(rf"Error vs. samples ($T={FIXED_TIME:g}$)")

    for ax in (ax_time, ax_samples):
        ax.set_yscale("log")
        ax.set_ylabel("Hausdorff distance", fontsize=26)
        ax.xaxis.label.set_size(26)
        ax.title.set_size(26)
        ax.tick_params(axis="both", which="major", labelsize=19)
        ax.tick_params(axis="y", which="minor", labelsize=14)
        ax.grid(True, which="major", color=GRID, linewidth=1.0)
        ax.set_axisbelow(True)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="re-plot from the saved trials CSV")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    args = parser.parse_args()
    use_serif_fonts("STIXGeneral")
    print(f"triangle: kappa={TRIANGLE_KAPPA:.4f}, lambda={TRIANGLE_LAMBDA:.4f}, diameter={TRIANGLE_DIAMETER:.4f}, "
          f"rho_min={RHO_MIN:.4f}, rho_max={RHO_MAX:.4f}")

    if args.plot_only:
        rows = read_csv(TRIALS_CSV)
    else:
        rows = run_experiment(args.workers)
        write_csv(rows, TRIALS_CSV)
        print(f"saved {TRIALS_CSV}")
    stats = aggregate(rows)
    fits = fit_rates(stats)
    write_csv(fits, RATES_CSV)
    for fit in fits:
        print(f"{fit['system']:<9} mu={fit['mu']:+.2f} L={fit['lipschitz']:6.2f}: fitted rate {fit['fitted_rate']:+.3f}, "
              f"d_H(T=2)/d_H(T=0)={fit['error_ratio_T2_over_T0']:.3f} (flow Lipschitz constant "
              f"{fit['flow_lipschitz_T2']:.3f}), slope in N {fit['slope_in_N']:+.3f}")
    plot_figure(stats)
    print(f"saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
