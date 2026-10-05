"""Geometry: the local volume density kappa of the initial set changes the constant, not the rate.

Planar initial sets of equal area pi (R = 1) are sampled i.i.d. uniformly and
propagated through the spiral sink x' = M x, M = [[-1, -4], [4, -1]], to T = 1,
so phi_T = e^{-T} Rot(4T) and the one-sided Lipschitz constant is mu = -1.  The
estimator is the endpoint cloud itself, the smallest estimator that contains
the samples, and the error is its exact Hausdorff distance to the reachable set.

    sample sweep   disk, square, isosceles triangles with apex angle 60, 30, 15
                   and 7.5 degrees (kappa = angle / (2 pi)), and a quadratic
                   cusp, which is not standard; N from 30 to 10^5
    angle sweep    triangles with apex angle from 60 down to 0.94 degrees at
                   N = 10^4

Every condition has 200 trials.  The theory curves are the upper bound for
standard and convex sets and the minimax lower bound for spikes, both solved
for r, and the mean of the distance from the apex to the cloud, whose law is
P(d > r) = (1 - kappa (e^{-mu T} r / R)^2)^N.

    python -m experiments.corner_angle_scaling              # run + plot
    python -m experiments.corner_angle_scaling --plot-only  # plot from CSV
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from reachapprox.bounds import certified_accuracy, lower_bound_accuracy
from reachapprox.flows import spiral_flow
from reachapprox.geometry import (
    CENTER,
    DISK_RADIUS,
    TARGET_AREA,
    contains_points,
    convex_polygon_standardness,
    dense_boundary,
    isosceles_triangle,
    sample_disk,
    sample_uniform_polygon,
    square_set,
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
RATE = -1.0
OMEGA = 4.0
HORIZON = 1.0
RHO = 1.0
DELTA = 0.05
COMMON_R = 1.0

SAMPLE_SIZES = (30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000)
ANGLE_SWEEP_SAMPLE_SIZE = 10_000
ANGLE_SWEEP_DEG = tuple(60.0 * 2.0 ** (-k / 2.0) for k in range(13))
N_TRIALS = 200
RANDOM_SEED = 23
BOUNDARY_SPACING = 2e-4
SLOPE_FIT_MIN_N = 1_000
SLOPE_FIT_MAX_ANGLE_DEG = 7.5

CUSP_EXPONENT = 2.0
CUSP_LENGTH = 3.0
CUSP_HALF_WIDTH = TARGET_AREA * (CUSP_EXPONENT + 1.0) / (2.0 * CUSP_LENGTH)
CUSP_TIP = CENTER - np.array([CUSP_LENGTH * (CUSP_EXPONENT + 1.0) / (CUSP_EXPONENT + 2.0), 0.0])

# (kind, apex angle in degrees) of the sets in the sample sweep; the angle is 0 for non-triangles.
SAMPLE_SWEEP_SETS = (("disk", 0.0), ("square", 0.0), ("triangle", 60.0), ("triangle", 30.0),
                     ("triangle", 15.0), ("triangle", 7.5), ("cusp", 0.0))

OUT_DIR = RESULTS_DIR / "corner_angle_scaling"
TRIALS_CSV = OUT_DIR / "trials.csv"
SLOPES_CSV = OUT_DIR / "loglog_slopes.csv"
FIGURE_PATH = OUT_DIR / "corner_angle_scaling.png"

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
TRIANGLE_RAMP = {60.0: "#86b6ef", 30.0: "#3987e5", 15.0: "#1c5cab", 7.5: "#0d366b"}
STYLES = {
    ("disk", 0.0): {"label": "disk", "color": "#eb6834", "marker": "o", "linestyle": "-"},
    ("square", 0.0): {"label": "square", "color": "#1baf7a", "marker": "s", "linestyle": "-"},
    ("cusp", 0.0): {"label": "cusp", "color": MUTED, "marker": "X", "linestyle": "--"},
    **{("triangle", angle): {"label": rf"${angle:g}^\circ$", "color": color, "marker": "^", "linestyle": "-"}
       for angle, color in TRIANGLE_RAMP.items()},
}
# Lower-left corner x and center y of each outline in the display panel, which doubles as the legend.
SET_LANES = {("disk", 0.0): (0.0, 9.6), ("square", 0.0): (3.2, 9.6), ("cusp", 0.0): (6.0, 9.6),
             ("triangle", 60.0): (0.0, 6.0), ("triangle", 30.0): (4.2, 6.0),
             ("triangle", 15.0): (0.0, 3.4), ("triangle", 7.5): (0.0, 1.2)}


@dataclass(frozen=True)
class InitialShape:
    sample: Callable[[np.random.Generator, int], np.ndarray]
    contains: Callable[[np.ndarray], np.ndarray]
    boundary: np.ndarray
    corner: np.ndarray  # boundary point with the smallest local volume


def cusp_polygon(n_points: int = 2_000) -> Polygon:
    """Polygonal outline of the cusp {0 < u < a, |v| < b (u / a)^gamma}, tip at CUSP_TIP."""
    s = np.linspace(0.0, 1.0, n_points) ** 2
    upper = np.column_stack((CUSP_LENGTH * s, CUSP_HALF_WIDTH * s**CUSP_EXPONENT))
    return Polygon(CUSP_TIP + np.vstack((upper, (upper * [1.0, -1.0])[-1:0:-1])))


def sample_cusp(rng: np.random.Generator, n: int) -> np.ndarray:
    u = rng.random(n) ** (1.0 / (CUSP_EXPONENT + 1.0))
    v = CUSP_HALF_WIDTH * u**CUSP_EXPONENT * rng.uniform(-1.0, 1.0, n)
    return CUSP_TIP + np.column_stack((CUSP_LENGTH * u, v))


def cusp_contains(points: np.ndarray) -> np.ndarray:
    u = (points[:, 0] - CUSP_TIP[0]) / CUSP_LENGTH
    inside = (u >= 0.0) & (u <= 1.0)
    return inside & (np.abs(points[:, 1] - CUSP_TIP[1]) <= CUSP_HALF_WIDTH * np.where(inside, u, 0.0) ** CUSP_EXPONENT)


def polygon_of(kind: str, angle_deg: float) -> Polygon:
    return square_set() if kind == "square" else isosceles_triangle(np.radians(angle_deg))


def make_shape(kind: str, angle_deg: float) -> InitialShape:
    if kind == "disk":
        angles = np.arange(0.0, 2.0 * np.pi, BOUNDARY_SPACING / DISK_RADIUS)
        boundary = CENTER + DISK_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))
        return InitialShape(sample_disk, lambda p: np.sum((p - CENTER) ** 2, axis=1) <= DISK_RADIUS**2,
                            boundary, boundary[0])
    if kind == "cusp":
        return InitialShape(sample_cusp, cusp_contains, dense_boundary(cusp_polygon(), BOUNDARY_SPACING), CUSP_TIP)
    geom = polygon_of(kind, angle_deg)
    return InitialShape(lambda rng, n: sample_uniform_polygon(rng, geom, n), lambda p: contains_points(geom, p),
                        dense_boundary(geom, BOUNDARY_SPACING), np.asarray(geom.exterior.coords[0], dtype=float))


def theory_parameters(kind: str, angle_deg: float) -> dict:
    """Constants of the upper bound: (kappa, lam) = (2^-n, r0) for the disk, (kappa_P, h_P) for polygons."""
    if kind == "disk":
        kappa, lam, diameter = 2.0**-N_DIM, DISK_RADIUS, 2.0 * DISK_RADIUS
    else:
        geom = polygon_of(kind, angle_deg)
        kappa, lam = convex_polygon_standardness(geom)
        vertices = np.asarray(geom.exterior.coords[:-1])
        diameter = float(np.max(np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2)))
    return {"n": N_DIM, "mu": RATE, "R": COMMON_R, "kappa": kappa, "lam": lam, "diameter": diameter,
            "convex": True, "rho": RHO, "delta": DELTA}


def flow(points: np.ndarray) -> np.ndarray:
    return spiral_flow(points, HORIZON, RATE, OMEGA)


def trial_seed(kind: str, angle_deg: float, sweep: str, sample_size: int, trial: int) -> int:
    return RANDOM_SEED + stable_seed_offset(kind, f"corner-{sweep}", angle_deg, sample_size) + trial


def run_condition(task: tuple[str, float, str, int]) -> list[dict]:
    """All trials of one (initial set, sweep, N) condition."""
    kind, angle_deg, sweep, sample_size = task
    shape = make_shape(kind, angle_deg)
    boundary, corner = flow(shape.boundary), flow(shape.corner[None, :])[0]

    def contains(points: np.ndarray) -> np.ndarray:
        return shape.contains(spiral_flow(points, -HORIZON, RATE, OMEGA))

    rows = []
    for trial in range(N_TRIALS):
        seed = trial_seed(kind, angle_deg, sweep, sample_size, trial)
        endpoints = flow(shape.sample(np.random.default_rng(seed), sample_size))
        rows.append({"sweep": sweep, "set": kind, "angle_deg": angle_deg, "T": HORIZON, "N": sample_size,
                     "trial": trial, "seed": seed, "error": cloud_inner_error(endpoints, boundary, contains),
                     "corner_distance": float(np.min(np.linalg.norm(endpoints - corner, axis=1)))})
    return rows


def run_experiment(workers: int | None) -> list[dict]:
    tasks = [(kind, angle, "sample", n) for kind, angle in SAMPLE_SWEEP_SETS for n in SAMPLE_SIZES]
    tasks += [("triangle", angle, "angle", ANGLE_SWEEP_SAMPLE_SIZE) for angle in ANGLE_SWEEP_DEG]
    tasks.sort(key=lambda task: -task[3])  # longest conditions first
    rows = []
    for (kind, angle, sweep, n), task_rows in zip(tasks, parallel_map(run_condition, tasks, workers)):
        mean = np.mean([row["error"] for row in task_rows])
        print(f"{sweep:<6} {kind:<8} angle={angle:7.3f} N={n:>6}: mean d_H={mean:.5f}", flush=True)
        rows += task_rows
    return rows


def aggregate(rows: list[dict]) -> dict:
    """(sweep, set, angle, N) -> mean, 95% CI of the mean, 5% and 95% quantiles, and mean corner distance."""
    grouped: dict[tuple, list[tuple[float, float]]] = {}
    for row in rows:
        key = (row["sweep"], row["set"], round(float(row["angle_deg"]), 6), int(float(row["N"])))
        grouped.setdefault(key, []).append((float(row["error"]), float(row["corner_distance"])))
    stats = {}
    for key, values in grouped.items():
        errors, corners = np.array(values).T
        mean, ci_low, ci_high = (float(v) for v in mean_ci95(errors))
        stats[key] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "q05": np.quantile(errors, 0.05),
                      "q95": np.quantile(errors, 0.95), "corner_mean": corners.mean()}
    return stats


def fit_slopes(stats: dict) -> list[dict]:
    """Least-squares log-log slopes of the error, in N for each set and in kappa for the thin triangles."""
    fits = []
    sizes = [n for n in SAMPLE_SIZES if n >= SLOPE_FIT_MIN_N]
    for kind, angle in SAMPLE_SWEEP_SETS:
        means = [stats[("sample", kind, angle, n)]["mean"] for n in sizes]
        predicted = -1.0 / (1.0 + CUSP_EXPONENT) if kind == "cusp" else -1.0 / N_DIM
        fits.append({"set": kind, "angle_deg": angle, "quantity": "mean", "variable": "N",
                     "fit_range": f"N>={SLOPE_FIT_MIN_N}", "slope": np.polyfit(np.log(sizes), np.log(means), 1)[0],
                     "predicted": predicted})
    thin = [angle for angle in ANGLE_SWEEP_DEG if angle <= SLOPE_FIT_MAX_ANGLE_DEG]
    kappas = [angle / 360.0 for angle in thin]
    for quantity in ("mean", "q95", "corner_mean"):
        values = [stats[("angle", "triangle", round(angle, 6), ANGLE_SWEEP_SAMPLE_SIZE)][quantity] for angle in thin]
        fits.append({"set": "triangle", "angle_deg": f"<={SLOPE_FIT_MAX_ANGLE_DEG:g}", "quantity": quantity,
                     "variable": "kappa", "fit_range": f"N={ANGLE_SWEEP_SAMPLE_SIZE}",
                     "slope": np.polyfit(np.log(kappas), np.log(values), 1)[0], "predicted": -1.0 / N_DIM})
    return fits


def apex_distance_mean(kappa, sample_size) -> np.ndarray:
    """E d(phi_T(apex), cloud) ~ e^{mu T} R Gamma(3/2) / sqrt(kappa N), from the law in the module docstring."""
    return np.exp(RATE * HORIZON) * COMMON_R * np.sqrt(np.pi / (4.0 * np.asarray(kappa) * sample_size))


def draw_sets(ax) -> None:
    """Outlines of the initial sets, each with the marker of its curve in the error panels."""
    for key, (x0, y0) in SET_LANES.items():
        kind, angle = key
        style = STYLES[key]
        if kind == "disk":
            outline = make_shape(kind, angle).boundary
        else:
            outline = np.asarray((cusp_polygon() if kind == "cusp" else polygon_of(kind, angle)).exterior.coords)
        shifted = outline - [outline[:, 0].min(), 0.5 * (outline[:, 1].min() + outline[:, 1].max())] + [x0, y0]
        ax.fill(shifted[:, 0], shifted[:, 1], facecolor=style["color"], alpha=0.16, edgecolor="none")
        ax.plot(*np.vstack((shifted, shifted[:1])).T, color=style["color"], linewidth=2.2, linestyle=style["linestyle"])
        if kind == "triangle":  # key to the right of the apex
            key_x, key_y = shifted[:, 0].max() + 0.45, y0
        else:  # key above the set
            key_x, key_y = x0 + 0.2, shifted[:, 1].max() + 0.5
        ax.plot([key_x], [key_y], marker=style["marker"], color=style["color"], markersize=11,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(key_x + 0.3, key_y, style["label"], fontsize=20, color=INK, ha="left", va="center")
    ax.text(9.5, 2.35, "isosceles triangles,\n" r"apex angle $\theta$", fontsize=19, color=MUTED, ha="right",
            va="center", linespacing=1.25)
    ax.set_xlim(-0.5, 9.8)
    ax.set_ylim(0.3, 12.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"Initial sets (area $\pi$)")


def plot_figure(stats: dict) -> None:
    fig, (ax_sets, ax_samples, ax_angle) = plt.subplots(
        1, 3, figsize=(23.5, 7.4), gridspec_kw={"width_ratios": (0.85, 1.15, 1.15), "wspace": 0.22})
    draw_sets(ax_sets)

    sizes = np.asarray(SAMPLE_SIZES, dtype=float)
    for key in SAMPLE_SWEEP_SETS:
        style = STYLES[key]
        mean, low, high = (np.array([stats[("sample", *key, n)][c] for n in SAMPLE_SIZES])
                           for c in ("mean", "ci_low", "ci_high"))
        ax_samples.plot(sizes, mean, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                        markersize=10, markeredgecolor="white", markeredgewidth=1.2, linewidth=2.2)
        ax_samples.fill_between(sizes, low, high, color=style["color"], alpha=0.16, linewidth=0)
    for key, name, linestyle in ((("triangle", 7.5), r"$7.5^\circ$ triangle", "-."), (("square", 0.0), "square", ":")):
        bound = certified_accuracy(sizes, HORIZON, **theory_parameters(*key))
        ax_samples.plot(sizes, bound, color=INK, linestyle=linestyle, linewidth=2.4, label=f"upper bound, {name}")
    guide = np.array([30.0, 500.0])
    for exponent, label, offset in ((-1.0 / 3.0, r"$\propto N^{-1/3}$", 1.25), (-1.0 / N_DIM, r"$\propto N^{-1/2}$", 0.62)):
        values = 0.028 * (guide / guide[0]) ** exponent
        ax_samples.plot(guide, values, color=MUTED, linewidth=1.6)
        ax_samples.text(guide[1] * 1.12, values[1] * offset, label, fontsize=19, color=MUTED, ha="left", va="center")
    ax_samples.set_xscale("log")
    ax_samples.set_xticks(SAMPLE_SIZES)
    ax_samples.set_xticklabels(power_of_ten_labels(SAMPLE_SIZES), rotation=35, ha="right")
    ax_samples.minorticks_off()
    ax_samples.set_xlabel(r"sample size $N$")
    ax_samples.set_title(rf"Error vs. samples ($T={HORIZON:g}$)")
    ax_samples.legend(loc="upper right", frameon=False, fontsize=19, handlelength=2.4, labelspacing=0.4)

    angles = np.asarray(ANGLE_SWEEP_DEG)
    kappas = angles / 360.0
    sweep = {c: np.array([stats[("angle", "triangle", round(a, 6), ANGLE_SWEEP_SAMPLE_SIZE)][c] for a in angles])
             for c in ("mean", "q05", "q95", "corner_mean")}
    dense_kappa = np.geomspace(kappas.min() / 1.15, kappas.max() * 1.15, 200)
    upper = np.array([certified_accuracy(ANGLE_SWEEP_SAMPLE_SIZE, HORIZON, **theory_parameters("triangle", a))
                      for a in angles])
    lower = lower_bound_accuracy(ANGLE_SWEEP_SAMPLE_SIZE, HORIZON, n=N_DIM, mu=RATE, R=COMMON_R, rho=RHO,
                                 delta=DELTA, kappa=dense_kappa)
    color = TRIANGLE_RAMP[30.0]
    ax_angle.errorbar(kappas, sweep["mean"], yerr=(sweep["mean"] - sweep["q05"], sweep["q95"] - sweep["mean"]),
                      color=color, marker="^", markersize=10, markeredgecolor="white", markeredgewidth=1.2,
                      linewidth=2.2, capsize=4, label=r"$d_H$: mean, 5-95% range")
    ax_angle.plot(kappas, sweep["corner_mean"], color=color, marker="o", markersize=8.5, markerfacecolor="white",
                  markeredgewidth=1.8, linestyle="none", label="apex-to-cloud distance: mean")
    ax_angle.plot(dense_kappa, apex_distance_mean(dense_kappa, ANGLE_SWEEP_SAMPLE_SIZE), color=MUTED,
                  linewidth=1.6, label=r"apex law, $\propto\kappa^{-1/2}$")
    ax_angle.plot(kappas, upper, color=INK, linestyle="-.", linewidth=2.4, label="upper bound")
    ax_angle.plot(dense_kappa, lower, color=INK, linestyle="--", linewidth=2.2, label="minimax lower bound (spike)")
    ax_angle.set_xscale("log")
    ax_angle.set_xticks(kappas[::2])
    ax_angle.set_xticklabels([rf"$\frac{{1}}{{{360.0 / a:.0f}}}$" for a in angles[::2]])
    ax_angle.minorticks_off()
    top_axis = ax_angle.secondary_xaxis("top")
    top_axis.set_xscale("log")
    top_axis.set_xticks(kappas[::2])
    top_axis.set_xticklabels([rf"${a:.2g}^\circ$" for a in angles[::2]])
    top_axis.minorticks_off()
    top_axis.tick_params(labelsize=18)
    ax_angle.set_xlabel(r"local volume density $\kappa=\theta/(2\pi)$")
    ax_angle.set_title(rf"Error vs. apex angle $\theta$ ($N=10^{{{int(np.log10(ANGLE_SWEEP_SAMPLE_SIZE))}}}$)", pad=40)
    handles, labels = ax_angle.get_legend_handles_labels()
    order = [labels.index(label) for label in (r"$d_H$: mean, 5-95% range", "apex-to-cloud distance: mean",
                                                r"apex law, $\propto\kappa^{-1/2}$", "upper bound",
                                                "minimax lower bound (spike)")]
    ax_angle.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", frameon=False,
                    fontsize=17, handlelength=2.2, labelspacing=0.35)

    for ax in (ax_samples, ax_angle):
        ax.set_yscale("log")
        ax.set_ylabel("Hausdorff distance", fontsize=26)
        ax.xaxis.label.set_size(26)
        ax.tick_params(axis="both", which="major", labelsize=19)
        ax.tick_params(axis="x", which="major", labelsize=21 if ax is ax_angle else 19)
        ax.tick_params(axis="y", which="minor", labelsize=14)
        ax.grid(True, which="major", color=GRID, linewidth=1.0)
        ax.set_axisbelow(True)
    ax_angle.set_ylim(top=ax_angle.get_ylim()[1] * 3.0)  # room for the legend
    for ax in (ax_sets, ax_samples, ax_angle):
        ax.title.set_size(26)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="re-plot from the saved trials CSV")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    args = parser.parse_args()
    use_serif_fonts("STIXGeneral")

    if args.plot_only:
        rows = read_csv(TRIALS_CSV)
    else:
        rows = run_experiment(args.workers)
        write_csv(rows, TRIALS_CSV)
        print(f"saved {TRIALS_CSV}")
    stats = aggregate(rows)
    fits = fit_slopes(stats)
    write_csv(fits, SLOPES_CSV)
    for fit in fits:
        print(f"slope of {fit['quantity']:<11} in {fit['variable']:<5} {fit['set']:<8} angle={fit['angle_deg']!s:<5}: "
              f"{fit['slope']:+.3f} (predicted {fit['predicted']:+.3f})")
    plot_figure(stats)
    print(f"saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
