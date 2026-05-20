"""Christoffel estimator under uniform/adversarial mixture sampling.

Run from /home/jixia/exp with:

    .venv/bin/python -u reachapprox/exp/quaddynadv/quaddyn_christoffel_mixture_experiment.py

This script evaluates mixture ratios between uniform and adversarial endpoint
samples. For each sample budget N, it saves one figure with three panels
corresponding to the disk, triangle, and opened-triangle initial sets.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from reachapprox import utils as ru


MIXTURE_UNIFORM_FRACTIONS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
SAMPLE_BUDGETS = (10, 100, 1000)
TIME_GRID = tuple(float(t) for t in np.geomspace(0.01, 0.28, 13))
N_TRIALS = 50
RANDOM_SEED = 11
CHRISTOFFEL_REFERENCE_SAMPLES = 12_000

OUT_DIR = Path("reachapprox/exp/quaddynadv/results")
FIGURE_TEMPLATE = "hausdorff_vs_time_christoffel_mixture_N{budget}.png"

TITLE_SIZE = 22
SUBTITLE_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 12
LEGEND_SIZE = 11


def mixture_label(uniform_fraction: float) -> str:
    adv_fraction = 1.0 - uniform_fraction
    return f"{uniform_fraction:.1f}U/{adv_fraction:.1f}A"


def mixed_endpoint_cloud(
    uniform_endpoints: np.ndarray,
    adversarial_endpoints: np.ndarray,
    total_budget: int,
    uniform_fraction: float,
) -> np.ndarray:
    """Combine uniform and adversarial endpoint clouds with a fixed total budget."""
    n_uniform = int(round(uniform_fraction * total_budget))
    n_uniform = min(max(n_uniform, 0), total_budget)
    n_adversarial = total_budget - n_uniform

    pieces = []
    if n_uniform:
        pieces.append(uniform_endpoints[:n_uniform])
    if n_adversarial:
        pieces.append(adversarial_endpoints[:n_adversarial])
    return np.vstack(pieces)


def run_mixture_sweep_for_budget(
    initial_sets: list[ru.InitialSet],
    budget: int,
) -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return mean/95% CI curves for all initial sets and mixture ratios."""
    rng = np.random.default_rng(RANDOM_SEED + 210_000 + budget)
    reference_initial = {
        item.name: ru.sample_uniform_polygon(rng, item.geom, CHRISTOFFEL_REFERENCE_SAMPLES)
        for item in initial_sets
    }

    curves: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    print(f"\nRunning Christoffel mixture sweep for N={budget}:")

    for item in initial_sets:
        reference_clouds = {
            t: ru.flow(reference_initial[item.name], t)
            for t in TIME_GRID
        }
        values = {
            frac: np.empty((len(TIME_GRID), N_TRIALS), dtype=float)
            for frac in MIXTURE_UNIFORM_FRACTIONS
        }

        for t_index, t in enumerate(TIME_GRID):
            reference_points = reference_clouds[t]
            for trial in range(N_TRIALS):
                seed_offset = ru.stable_seed_offset(item.name, "christoffel-mixture", t, budget)
                uniform_rng = np.random.default_rng(RANDOM_SEED + 220_000 + seed_offset + trial)
                adv_rng = np.random.default_rng(RANDOM_SEED + 230_000 + seed_offset + trial)

                uniform_endpoints, _ = ru.run_uniform_sampling(uniform_rng, item.geom, t, budget)
                adversarial_endpoints, _ = ru.run_adversarial_sampling(adv_rng, item.geom, t, budget)

                for frac in MIXTURE_UNIFORM_FRACTIONS:
                    endpoints = mixed_endpoint_cloud(uniform_endpoints, adversarial_endpoints, budget, frac)
                    values[frac][t_index, trial] = ru.christoffel_support_hausdorff(
                        reference_points, endpoints
                    )

            means = ", ".join(
                f"{mixture_label(frac)}={values[frac][t_index].mean():.6f}"
                for frac in MIXTURE_UNIFORM_FRACTIONS
            )
            print(f"{item.label:<16} t={t:.4f}: {means}")

        for frac in MIXTURE_UNIFORM_FRACTIONS:
            mean = values[frac].mean(axis=1)
            stderr = values[frac].std(axis=1, ddof=1) / np.sqrt(N_TRIALS)
            ci = 1.96 * stderr
            lo = np.maximum(mean - ci, np.finfo(float).tiny)
            hi = mean + ci
            curves[(item.name, frac)] = (mean, lo, hi)

    return curves


def plot_mixture_curves(
    curves: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    initial_sets: list[ru.InitialSet],
    budget: int,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(MIXTURE_UNIFORM_FRACTIONS)))
    markers = ("o", "s", "D", "^", "v", "P")

    for ax, item in zip(axes, initial_sets):
        for frac, color, marker in zip(MIXTURE_UNIFORM_FRACTIONS, colors, markers):
            mean, lo, hi = curves[(item.name, frac)]
            ax.plot(
                TIME_GRID,
                mean,
                color=color,
                marker=marker,
                lw=2.0,
                ms=4.5,
                label=mixture_label(frac),
            )
            ax.fill_between(TIME_GRID, lo, hi, color=color, alpha=0.20)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(TIME_GRID)
        ax.set_xticklabels([f"{t:.4f}" for t in TIME_GRID], rotation=35, ha="right", fontsize=TICK_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.set_title(item.label, fontsize=SUBTITLE_SIZE)
        ax.set_xlabel("time t", fontsize=LABEL_SIZE)
        ax.grid(True, which="both", alpha=0.28)

    axes[0].set_ylabel("Hausdorff distance", fontsize=LABEL_SIZE)
    axes[-1].legend(frameon=False, fontsize=LEGEND_SIZE, loc="best", title="mixture")
    fig.suptitle(f"Christoffel reachable-set error, N={budget}", fontsize=TITLE_SIZE)

    figure_path = OUT_DIR / FIGURE_TEMPLATE.format(budget=budget)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    initial_sets = ru.build_equal_area_initial_sets()
    saved_paths = []

    for budget in SAMPLE_BUDGETS:
        curves = run_mixture_sweep_for_budget(initial_sets, budget)
        saved_paths.append(plot_mixture_curves(curves, initial_sets, budget))

    print("\nSaved figures:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
