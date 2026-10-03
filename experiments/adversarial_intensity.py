"""Figures 9 and 10 (Appendix C.3): adversarial sampling intensity across estimators.

Time sweep for x' = x^2, y' = 0 with the equal-area circle, opened triangle,
and triangle.  For each budget N in {10, 100, 1000} the endpoints come from
Algorithm 1 with n_adv in {0, ..., 4} adversarial updates (n_adv = 0 is
uniform sampling).  The reachable set is estimated by the convex hull
(Figure 9) or the Christoffel sublevel set (Figure 10), and the Hausdorff
error to a 12,000-point reference cloud is averaged over 50 trials.

    python -m experiments.adversarial_intensity
    python -m experiments.adversarial_intensity --estimators christoffel
    python -m experiments.adversarial_intensity --plot-only
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from reachapprox.adversarial import adversarial_endpoints
from reachapprox.estimators import convex_hull_polygon
from reachapprox.flows import quadratic_flow
from reachapprox.geometry import equal_area_initial_sets, sample_uniform_polygon
from reachapprox.metrics import christoffel_hausdorff, hull_hausdorff
from reachapprox.utils import (
    RESULTS_DIR,
    mean_ci95,
    parallel_map,
    read_csv,
    stable_seed_offset,
    use_serif_fonts,
    write_csv,
)


N_ADV_VALUES = (0, 1, 2, 3, 4)
SAMPLE_BUDGETS = (10, 100, 1000)
TIME_GRID = tuple(float(t) for t in np.linspace(0.01, 0.29, 15))
N_TRIALS = 50
RANDOM_SEED = 11
REFERENCE_SAMPLES = 12_000

# Per-estimator seed bases and labels; the two sweeps use independent streams.
ESTIMATORS = {
    "convex_hull": {"title": "Convex-hull", "reference_seed": 310_000, "trial_seed": 330_000, "tag": "hull"},
    "christoffel": {"title": "Christoffel", "reference_seed": 210_000, "trial_seed": 230_000, "tag": "christoffel"},
}

OUT_DIR = RESULTS_DIR / "adversarial_intensity"
TRIALS_CSV = OUT_DIR / "trials.csv"


def figure_path(estimator: str, budget: int):
    return OUT_DIR / f"{estimator}_N{budget}.png"


def run_condition(task: tuple[str, int, str, float]) -> list[dict]:
    """All n_adv values and trials for one (estimator, budget, initial set, time)."""
    estimator, budget, set_name, t = task
    spec = ESTIMATORS[estimator]
    initial_sets = equal_area_initial_sets()
    rng = np.random.default_rng(RANDOM_SEED + spec["reference_seed"] + budget)
    reference_initial = {item.name: sample_uniform_polygon(rng, item.geom, REFERENCE_SAMPLES) for item in initial_sets}
    geom = next(item.geom for item in initial_sets if item.name == set_name)
    reference = quadratic_flow(reference_initial[set_name], t)

    rows = []
    for n_adv in N_ADV_VALUES:
        offset = stable_seed_offset(set_name, f"{spec['tag']}-nadv-{n_adv}", t, budget)
        for trial in range(N_TRIALS):
            trial_rng = np.random.default_rng(RANDOM_SEED + spec["trial_seed"] + offset + trial)
            endpoints = adversarial_endpoints(trial_rng, geom, t, budget, n_adv)
            if estimator == "convex_hull":
                error = hull_hausdorff(trial_rng, reference, convex_hull_polygon(endpoints))
            else:
                error = christoffel_hausdorff(reference, endpoints)
            rows.append({"estimator": estimator, "set": set_name, "N": budget, "n_adv": n_adv, "t": t,
                         "trial": trial, "error": error})
    return rows


def run_experiment(estimators: list[str], initial_sets, workers: int | None) -> list[dict]:
    tasks = [(estimator, budget, item.name, t) for estimator in estimators for budget in SAMPLE_BUDGETS
             for item in initial_sets for t in TIME_GRID]
    rows = []
    for (estimator, budget, set_name, t), task_rows in zip(tasks, parallel_map(run_condition, tasks, workers)):
        means = [np.mean([r["error"] for r in task_rows if r["n_adv"] == n_adv]) for n_adv in N_ADV_VALUES]
        print(f"{estimator:<11} N={budget:<5} {set_name:<8} t={t:.4f}  "
              + ", ".join(f"n_adv={n}: {m:.4f}" for n, m in zip(N_ADV_VALUES, means)))
        rows += task_rows
    return rows


def plot_budget(rows: list[dict], estimator: str, budget: int, initial_sets) -> None:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        if row["estimator"] == estimator and int(row["N"]) == budget:
            grouped.setdefault((row["set"], int(row["n_adv"]), float(row["t"])), []).append(float(row["error"]))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(N_ADV_VALUES)))
    for ax, item in zip(axes, initial_sets):
        for n_adv, color, marker in zip(N_ADV_VALUES, colors, ("o", "s", "D", "^", "v")):
            values = np.array([grouped[(item.name, n_adv, t)] for t in TIME_GRID])
            mean, lo, hi = mean_ci95(values, axis=1)
            ax.plot(TIME_GRID, mean, color=color, marker=marker, lw=2.0, ms=4.5, label=f"$n_{{adv}}={n_adv}$")
            ax.fill_between(TIME_GRID, lo, hi, color=color, alpha=0.20)
        ax.set_yscale("log")
        ax.set_xticks(TIME_GRID)
        ax.set_xticklabels([f"{t:.4f}" for t in TIME_GRID], rotation=35, ha="right", fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title(item.label, fontsize=18)
        ax.set_xlabel("time t", fontsize=18)
        ax.grid(True, which="both", alpha=0.28)
    axes[0].set_ylabel("Hausdorff distance", fontsize=18)
    axes[-1].legend(frameon=False, fontsize=11, loc="best", title=r"$n_{adv}$")
    fig.suptitle(f"{ESTIMATORS[estimator]['title']} reachable-set error, N={budget}", fontsize=22)
    fig.savefig(figure_path(estimator, budget), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="re-plot from the saved trials CSV")
    parser.add_argument("--estimators", default="convex_hull,christoffel")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    args = parser.parse_args()
    use_serif_fonts("DejaVu Serif")
    initial_sets = equal_area_initial_sets()
    estimators = [name for name in args.estimators.split(",") if name]

    if args.plot_only:
        rows = read_csv(TRIALS_CSV)
    else:
        rows = run_experiment(estimators, initial_sets, args.workers)
        write_csv(rows, TRIALS_CSV)
        print(f"saved {TRIALS_CSV}")
    for name in estimators:
        for budget in SAMPLE_BUDGETS:
            plot_budget(rows, name, budget, initial_sets)
            print(f"saved {figure_path(name, budget)}")


if __name__ == "__main__":
    main()
