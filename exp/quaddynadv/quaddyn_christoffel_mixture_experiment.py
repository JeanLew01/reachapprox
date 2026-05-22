"""Christoffel estimator under different adversarial-update counts.

Run from /home/jixia/exp with:

    .venv/bin/python -u reachapprox/exp/quaddynadv/quaddyn_christoffel_mixture_experiment.py

This script evaluates n_adv values, where larger n_adv implicitly increases the
adversarial share of the endpoint cloud. For each sample budget N, it saves one
figure with three panels
corresponding to the disk, triangle, and opened-triangle initial sets.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from reachapprox import utils as ru
from reachapprox.utils.quaddyn_sampling import ETA, adversarial_gradient


N_ADV_VALUES = (0, 1, 2, 3, 4)
SAMPLE_BUDGETS = (10, 100, 1000)
TIME_GRID = tuple(float(t) for t in np.geomspace(0.01, 0.28, 13))
N_TRIALS = 50
RANDOM_SEED = 11
CHRISTOFFEL_REFERENCE_SAMPLES = 12_000

OUT_DIR = Path("reachapprox/exp/quaddynadv/results")
FIGURE_TEMPLATE = "hausdorff_vs_time_christoffel_nadv_N{budget}.png"

TITLE_SIZE = 22
SUBTITLE_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 12
LEGEND_SIZE = 11


def n_adv_label(n_adv: int) -> str:
    return f"$n_{{adv}}={n_adv}$"


def adversarial_endpoint_cloud_with_budget(
    rng: np.random.Generator,
    geom,
    t: float,
    total_budget: int,
    n_adv: int,
) -> np.ndarray:
    """Run n_adv adversarial updates and return exactly total_budget endpoints."""
    if total_budget % (n_adv + 1) == 0:
        endpoints, _ = ru.run_adversarial_sampling(rng, geom, t, total_budget, n_adv=n_adv)
        return endpoints

    m_particles = int(np.ceil(total_budget / (n_adv + 1)))
    particles = ru.sample_uniform_polygon(rng, geom, m_particles)
    endpoint_batches = [ru.flow(particles, t)]

    for _ in range(n_adv):
        accumulated = np.vstack(endpoint_batches)
        c, Q = ru.endpoint_center_and_Q(accumulated)
        grad = adversarial_gradient(particles, t, c, Q)
        particles = ru.project_points_to_set(particles + ETA * grad, geom)
        endpoint_batches.append(ru.flow(particles, t))

    return np.vstack(endpoint_batches)[:total_budget]


def run_n_adv_sweep_for_budget(
    initial_sets: list[ru.InitialSet],
    budget: int,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return mean/95% CI curves for all initial sets and n_adv values."""
    rng = np.random.default_rng(RANDOM_SEED + 210_000 + budget)
    reference_initial = {
        item.name: ru.sample_uniform_polygon(rng, item.geom, CHRISTOFFEL_REFERENCE_SAMPLES)
        for item in initial_sets
    }

    curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    print(f"\nRunning Christoffel n_adv sweep for N={budget}:")

    for item in initial_sets:
        reference_clouds = {
            t: ru.flow(reference_initial[item.name], t)
            for t in TIME_GRID
        }
        values = {
            n_adv: np.empty((len(TIME_GRID), N_TRIALS), dtype=float)
            for n_adv in N_ADV_VALUES
        }

        for t_index, t in enumerate(TIME_GRID):
            reference_points = reference_clouds[t]
            for trial in range(N_TRIALS):
                for n_adv in N_ADV_VALUES:
                    seed_offset = ru.stable_seed_offset(
                        item.name, f"christoffel-nadv-{n_adv}", t, budget
                    )
                    adv_rng = np.random.default_rng(RANDOM_SEED + 230_000 + seed_offset + trial)
                    endpoints = adversarial_endpoint_cloud_with_budget(
                        adv_rng, item.geom, t, budget, n_adv
                    )
                    values[n_adv][t_index, trial] = ru.christoffel_support_hausdorff(
                        reference_points, endpoints
                    )

            means = ", ".join(
                f"n_adv={n_adv}: {values[n_adv][t_index].mean():.6f}"
                for n_adv in N_ADV_VALUES
            )
            print(f"{item.label:<16} t={t:.4f}: {means}")

        for n_adv in N_ADV_VALUES:
            mean = values[n_adv].mean(axis=1)
            stderr = values[n_adv].std(axis=1, ddof=1) / np.sqrt(N_TRIALS)
            ci = 1.96 * stderr
            lo = np.maximum(mean - ci, np.finfo(float).tiny)
            hi = mean + ci
            curves[(item.name, n_adv)] = (mean, lo, hi)

    return curves


def plot_n_adv_curves(
    curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    initial_sets: list[ru.InitialSet],
    budget: int,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(N_ADV_VALUES)))
    markers = ("o", "s", "D", "^", "v")

    for ax, item in zip(axes, initial_sets):
        for n_adv, color, marker in zip(N_ADV_VALUES, colors, markers):
            mean, lo, hi = curves[(item.name, n_adv)]
            ax.plot(
                TIME_GRID,
                mean,
                color=color,
                marker=marker,
                lw=2.0,
                ms=4.5,
                label=n_adv_label(n_adv),
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
    axes[-1].legend(frameon=False, fontsize=LEGEND_SIZE, loc="best", title=r"$n_{adv}$")
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
        curves = run_n_adv_sweep_for_budget(initial_sets, budget)
        saved_paths.append(plot_n_adv_curves(curves, initial_sets, budget))

    print("\nSaved figures:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
