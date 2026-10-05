"""Contraction: the three regimes of the sample complexity along the horizon.

Two initial sets of area pi (R = 1) are sampled i.i.d. uniformly and propagated
through the spiral sink x' = M x, M = [[-1, -4], [4, -1]] (mu = -1):

    square   convex, (1/4, 0.886)-standard
    array    nine squares of side 0.59 on a lattice of pitch 8 sides,
             (1/4, 0.295)-standard and far from convex

For a fixed accuracy r = 0.03, the empirical sample complexity N(T) is the 95%
quantile over 200 trials of the first N for which the Hausdorff distance
between the reachable set and the first N endpoints is at most r.  The flow is
a similarity, phi_T = e^{-T} Rot(4T), so that distance is e^{-T} times the
distance between the initial set and the first N initial states; one error
curve per trial therefore gives the hitting times for every horizon.

The theory curves are the upper bound in its three regimes (fine accuracy,
saturation at the regularity scale lambda, one sample once r >= diam(S_T)),
with the convex refinement for the square, and the minimax lower bound.

    python -m experiments.contraction_regimes              # run + plot
    python -m experiments.contraction_regimes --plot-only  # plot from CSV
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from reachapprox.bounds import lower_bound_samples, required_samples
from reachapprox.geometry import (
    CENTER,
    contains_points,
    dense_boundary,
    sample_uniform_set,
    square_array_set,
    square_set,
)
from reachapprox.metrics import cloud_inner_error
from reachapprox.utils import RESULTS_DIR, parallel_map, read_csv, stable_seed_offset, use_serif_fonts, write_csv


N_DIM = 2
RATE = -1.0
RHO = 1.0
DELTA = 0.05
COMMON_R = 1.0
ACCURACY = 0.03

ARRAY_PER_SIDE = 3
ARRAY_PITCH_RATIO = 8.0
HORIZONS = tuple(float(t) for t in np.arange(0.0, 6.76, 0.25))
N_TRIALS = 200
RANDOM_SEED = 47
MAX_SAMPLES = 2**17
SAMPLE_GRID = tuple(sorted(set(range(1, 65)) | {int(round(64 * 2 ** (j / 8))) for j in range(1, 8 * 11 + 1)}))
BOUNDARY_SPACING = 5e-4

OUT_DIR = RESULTS_DIR / "contraction_regimes"
TRIALS_CSV = OUT_DIR / "trials.csv"
SUMMARY_CSV = OUT_DIR / "sample_complexity.csv"
FIGURE_PATH = OUT_DIR / "contraction_regimes.png"

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
STYLES = {
    "square": {"label": "square (convex)", "color": "#2a78d6", "marker": "s"},
    "array": {"label": "array of nine squares", "color": "#eb6834", "marker": "o"},
}


def initial_set(name: str):
    return square_set() if name == "square" else square_array_set(ARRAY_PER_SIDE, ARRAY_PITCH_RATIO)


def theory_parameters(name: str) -> dict:
    """Standardness constants of a union of squares: kappa = 1/4 and lambda = half a side."""
    geom = initial_set(name)
    minx, miny, maxx, maxy = geom.bounds
    pieces = 1 if name == "square" else ARRAY_PER_SIDE**2
    return {"n": N_DIM, "mu": RATE, "R": COMMON_R, "kappa": 0.25, "lam": 0.5 * np.sqrt(geom.area / pieces),
            "diameter": float(np.hypot(maxx - minx, maxy - miny)), "convex": name == "square", "rho": RHO,
            "delta": DELTA}


def run_trial(task: tuple[str, int]) -> list[dict]:
    """d_H(S_0, X_{1:N}) along one i.i.d. sequence, for N on the grid until it drops below the accuracy."""
    name, trial = task
    geom = initial_set(name)
    boundary = dense_boundary(geom, BOUNDARY_SPACING)
    seed = RANDOM_SEED + stable_seed_offset(name, "contraction", 0.0, 0) + trial
    samples = sample_uniform_set(np.random.default_rng(seed), geom, MAX_SAMPLES)
    rows = []
    for n in SAMPLE_GRID:
        error = cloud_inner_error(samples[:n], boundary, lambda points: contains_points(geom, points))
        rows.append({"set": name, "trial": trial, "seed": seed, "N": n, "initial_error": error})
        if error <= ACCURACY:
            break
    return rows


def run_experiment(workers: int | None) -> list[dict]:
    tasks = [(name, trial) for name in STYLES for trial in range(N_TRIALS)]
    rows = []
    for (name, trial), trial_rows in zip(tasks, parallel_map(run_trial, tasks, workers)):
        last = trial_rows[-1]
        print(f"{name:<6} trial {trial:>3}: d_H <= {ACCURACY} at N={last['N']}"
              if last["initial_error"] <= ACCURACY else f"{name:<6} trial {trial:>3}: not reached", flush=True)
        rows += trial_rows
    return rows


def hitting_times(rows: list[dict]) -> dict[str, np.ndarray]:
    """set -> array (trial, horizon) of the first N with e^{mu T} d_H(S_0, X_{1:N}) <= accuracy (inf if none)."""
    curves: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in rows:
        curves.setdefault((row["set"], int(row["trial"])), []).append((int(row["N"]), float(row["initial_error"])))
    thresholds = ACCURACY * np.exp(-RATE * np.asarray(HORIZONS))
    times: dict[str, list[np.ndarray]] = {}
    for (name, _), curve in sorted(curves.items()):
        sizes, errors = np.array(sorted(curve)).T
        reached = errors[None, :] <= thresholds[:, None]
        first = np.where(reached.any(axis=1), sizes[np.argmax(reached, axis=1)], np.inf)
        times.setdefault(name, []).append(first)
    return {name: np.vstack(values) for name, values in times.items()}


def summarize(times: dict[str, np.ndarray]) -> list[dict]:
    horizons = np.asarray(HORIZONS)
    summary = []
    for name, values in times.items():
        parameters = theory_parameters(name)
        upper = required_samples(ACCURACY, horizons, **parameters)
        lower = lower_bound_samples(ACCURACY, horizons, n=N_DIM, mu=RATE, R=COMMON_R, rho=RHO, delta=DELTA)
        quantile = np.quantile(values, 1.0 - DELTA, axis=0, method="higher")
        for index, T in enumerate(HORIZONS):
            summary.append({"set": name, "T": T, "preimage_accuracy": ACCURACY * np.exp(-RATE * T),
                            "N_empirical": quantile[index], "N_median": np.median(values[:, index]),
                            "N_upper_bound": upper[index], "N_lower_bound": lower[index]})
    return summary


def regime_boundaries(name: str) -> tuple[float, float]:
    """Horizons at which the bound saturates (r = 2 e^{mu T} lambda) and becomes trivial (r = e^{mu T} diam)."""
    parameters = theory_parameters(name)
    return (float(np.log(2.0 * parameters["lam"] / ACCURACY) / -RATE),
            float(np.log(parameters["diameter"] / ACCURACY) / -RATE))


def draw_sets(ax) -> None:
    """The two initial sets to scale, with the preimage accuracy r e^{-mu T} at three horizons."""
    array = initial_set("array")
    for piece in array.geoms:
        ax.fill(*piece.exterior.xy, facecolor=STYLES["array"]["color"], alpha=0.85, edgecolor="none")
    square = np.asarray(square_set().exterior.coords) + [0.0, 8.6]
    ax.fill(square[:, 0], square[:, 1], facecolor=STYLES["square"]["color"], alpha=0.85, edgecolor="none")
    ax.text(CENTER[0] + 1.3, 8.6, "square", fontsize=20, color=INK, ha="left", va="center")
    ax.text(CENTER[0], 5.75, "array", fontsize=20, color=INK, ha="center", va="bottom")
    corner = np.array(array.geoms[0].exterior.coords[:-1]).mean(axis=0)
    for T, angle in ((3.5, 38.0), (5.0, 45.0), (5.5, 60.0)):
        radius = ACCURACY * np.exp(-RATE * T)
        arc = np.radians(np.linspace(-30.0, 120.0, 300))
        ax.plot(corner[0] + radius * np.cos(arc), corner[1] + radius * np.sin(arc), color=MUTED, linewidth=1.5,
                linestyle=(0, (4, 3)))
        offset = radius + (1.25 if T == 3.5 else 0.0)
        anchor = corner + offset * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        ax.text(*anchor, rf"$T={T:g}$", fontsize=16, color=INK, ha="center", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.plot(*corner, marker="o", color=INK, markersize=5)
    ax.set_xlim(-3.9, 7.9)
    ax.set_ylim(-5.9, 10.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Initial sets and\n" r"preimage accuracy $re^{-\mu T}$", linespacing=1.15)


def plot_figure(summary: list[dict]) -> None:
    fig, (ax_sets, ax) = plt.subplots(1, 2, figsize=(23.5, 7.4), gridspec_kw={"width_ratios": (0.62, 2.0),
                                                                             "wspace": 0.10})
    draw_sets(ax_sets)

    horizons = np.asarray(HORIZONS)
    dense = np.linspace(horizons[0], horizons[-1], 1_000)
    for name, style in STYLES.items():
        empirical = np.array([row["N_empirical"] for row in summary if row["set"] == name], dtype=float)
        ax.plot(dense, required_samples(ACCURACY, dense, **theory_parameters(name)), color=style["color"],
                linestyle="--", linewidth=2.0)
        ax.plot(horizons, empirical, color=style["color"], marker=style["marker"], markersize=10,
                markeredgecolor="white", markeredgewidth=1.2, linewidth=2.2, label=style["label"])
    lower = lower_bound_samples(ACCURACY, dense, n=N_DIM, mu=RATE, R=COMMON_R, rho=RHO, delta=DELTA)
    ax.plot(dense, lower, color=INK, linestyle="-.", linewidth=2.2, label="minimax lower bound")
    ax.plot([], [], color=INK, linestyle="--", linewidth=2.0, label="upper bound (same color)")
    guide = np.array([1.5, 2.6])
    ax.plot(guide, 25.0 * np.exp(N_DIM * RATE * (guide - guide[0])), color=MUTED, linewidth=1.6)
    ax.text(guide[1] + 0.06, 25.0 * np.exp(N_DIM * RATE * (guide[1] - guide[0])), r"$\propto e^{n\mu T}$",
            fontsize=20, color=MUTED, ha="left", va="center")

    saturation, trivial = regime_boundaries("array")
    for boundary in (saturation, trivial):
        ax.axvline(boundary, color=MUTED, linewidth=1.2, linestyle=":")
    y_text = 2.4e6
    for x, text in ((0.5 * saturation, "fine accuracy"), (0.5 * (saturation + trivial), r"saturation at scale $\lambda$"),
                    (0.5 * (trivial + horizons[-1]), "trivial")):
        ax.text(x, y_text, text, fontsize=19, color=INK, ha="center", va="center")
    ax.text(0.5 * (saturation + trivial), 6.5e5, "(regimes of the upper bound for the array)", fontsize=16,
            color=MUTED, ha="center", va="center")

    ax.set_yscale("log")
    ax.set_ylim(0.7, 6e6)
    ax.set_xlim(horizons[0] - 0.1, horizons[-1] + 0.1)
    ax.set_xlabel(r"horizon $T$", fontsize=26)
    ax.set_ylabel(rf"samples for $d_H\leq{ACCURACY:g}$ w.p. $0.95$", fontsize=26)
    ax.set_title(r"Sample complexity vs. horizon under contraction ($\mu=-1$)", fontsize=26)
    ax.tick_params(axis="both", which="major", labelsize=19)
    ax.grid(True, which="major", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", frameon=False, fontsize=19, handlelength=2.4, labelspacing=0.4,
              bbox_to_anchor=(1.0, 0.60))
    ax_sets.title.set_size(24)
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
    summary = summarize(hitting_times(rows))
    write_csv(summary, SUMMARY_CSV)
    for name in STYLES:
        saturation, trivial = regime_boundaries(name)
        print(f"{name}: bound saturates at T={saturation:.2f} and is trivial from T={trivial:.2f}")
        for row in summary:
            if row["set"] == name and row["T"] in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                print(f"  T={row['T']:.2f}: empirical N={row['N_empirical']:.0f}, upper bound {row['N_upper_bound']:.0f}")
    plot_figure(summary)
    print(f"saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
