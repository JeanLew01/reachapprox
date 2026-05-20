"""Uniform/adversarial sampling experiments for dx/dt=x^2, dy/dt=0.

Run from /home/jixia/exp with:

    .venv/bin/python -u reachapprox/exp/quaddynadv/quaddyn_adv_experiment.py

Shared geometry, sampling, flow, support-estimator, and Hausdorff utilities live
in :mod:`reachapprox.utils`; this file only defines the experiment workflow and
figures for the quadratic non-Lipschitz dynamics example.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reachapprox import utils as ru


SAMPLE_BUDGETS = (100, 1000)
TIME_GRID = tuple(float(t) for t in np.geomspace(0.01, 0.28, 13))
TIME_SWEEP_TRIALS = 50
REFERENCE_SAMPLES = 60_000
CHRISTOFFEL_REFERENCE_SAMPLES = 12_000
RANDOM_SEED = 11

OUT_DIR = Path("reachapprox/exp/quaddynadv/results")
INITIAL_SETS_FIG = OUT_DIR / "initial_sets.png"
UNIFORM_REACHABLE_FIG = OUT_DIR / "reachable_uniform.png"
ADVERSARIAL_REACHABLE_FIG = OUT_DIR / "reachable_adversarial.png"
HAUSDORFF_FIG = OUT_DIR / "hausdorff_uniform_vs_adversarial.png"
HAUSDORFF_TIME_CI_FIG = OUT_DIR / "hausdorff_vs_time_uniform_adversarial_ci.png"
HAUSDORFF_TIME_CHRISTOFFEL_CI_FIG = OUT_DIR / "hausdorff_vs_time_uniform_adversarial_ci_christoffel.png"

TITLE_SIZE = 22
SUBTITLE_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 12
LEGEND_SIZE = 11

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)


def plot_initial_sets(initial_sets: list[ru.InitialSet]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    for item in initial_sets:
        ax.add_patch(ru.polygon_patch(item.geom, facecolor=item.color, alpha=0.24, edgecolor=item.color, lw=2.0))
        centroid = item.geom.centroid
        print(
            f"{item.label:<16} area={item.geom.area:.8f}, "
            f"centroid=({centroid.x:.8f}, {centroid.y:.8f}), min_x={ru.min_x(item.geom):.8f}"
        )

    ax.scatter([2.0], [0.0], color="black", s=42, zorder=5, label="common center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x", fontsize=LABEL_SIZE)
    ax.set_ylabel("y", fontsize=LABEL_SIZE)
    ax.set_title("Equal-area initial sets", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.savefig(INITIAL_SETS_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def transformed_boundary(geom, T: float, n_points: int = 4_000) -> np.ndarray:
    return ru.flow(ru.boundary_points(geom, n_points), T)


def plot_reachable_clouds(
    results: dict[tuple[str, str, int], tuple[np.ndarray, object]],
    initial_sets: list[ru.InitialSet],
    T: float,
    method: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    budget = max(SAMPLE_BUDGETS)
    for ax, item in zip(axes, initial_sets):
        endpoints, hull = results[(item.name, method, budget)]
        boundary = transformed_boundary(item.geom, T)
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=3, color=item.color, alpha=0.28, linewidths=0)
        ax.add_patch(ru.polygon_patch(hull, facecolor=item.color, alpha=0.12, edgecolor=item.color, lw=2.0))
        ax.plot(boundary[:, 0], boundary[:, 1], color="black", lw=1.2)
        ax.set_title(f"{item.label}, N={budget}", fontsize=SUBTITLE_SIZE)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("x(T)", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
    axes[0].set_ylabel("y(T)", fontsize=LABEL_SIZE)
    fig.suptitle(f"{method.capitalize()} sampling reachable-set estimates", fontsize=TITLE_SIZE)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_budget_hausdorff(
    errors: dict[tuple[str, str, int], float],
    initial_sets: list[ru.InitialSet],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    for item in initial_sets:
        for method, linestyle, marker in (("uniform", "-", "o"), ("adversarial", "--", "s")):
            ys = [errors[(item.name, method, n)] for n in SAMPLE_BUDGETS]
            ax.plot(
                SAMPLE_BUDGETS,
                ys,
                linestyle=linestyle,
                marker=marker,
                color=item.color,
                lw=2.0,
                ms=6,
                label=f"{item.label}, {method}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample budget N", fontsize=LABEL_SIZE)
    ax.set_ylabel("Hausdorff distance", fontsize=LABEL_SIZE)
    ax.set_title("Convex-hull reachable-set error", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.savefig(HAUSDORFF_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fixed_time_comparison(
    initial_sets: list[ru.InitialSet],
    T: float,
) -> tuple[dict[tuple[str, str, int], tuple[np.ndarray, object]], dict[tuple[str, str, int], float]]:
    rng = np.random.default_rng(RANDOM_SEED)
    reference_points = {
        item.name: ru.flow(ru.sample_uniform_polygon(rng, item.geom, REFERENCE_SAMPLES), T)
        for item in initial_sets
    }
    results: dict[tuple[str, str, int], tuple[np.ndarray, object]] = {}
    errors: dict[tuple[str, str, int], float] = {}

    print("\nRunning sampling comparison:")
    for item in initial_sets:
        for budget in SAMPLE_BUDGETS:
            uniform_rng = np.random.default_rng(RANDOM_SEED + 1000 + budget)
            adv_rng = np.random.default_rng(RANDOM_SEED + 2000 + budget)
            err_rng = np.random.default_rng(RANDOM_SEED + 3000 + budget)

            uniform_points, uniform_hull = ru.run_uniform_sampling(uniform_rng, item.geom, T, budget)
            adv_points, adv_hull = ru.run_adversarial_sampling(adv_rng, item.geom, T, budget)

            results[(item.name, "uniform", budget)] = (uniform_points, uniform_hull)
            results[(item.name, "adversarial", budget)] = (adv_points, adv_hull)
            errors[(item.name, "uniform", budget)] = ru.approximate_hausdorff(
                err_rng, reference_points[item.name], uniform_hull
            )
            errors[(item.name, "adversarial", budget)] = ru.approximate_hausdorff(
                err_rng, reference_points[item.name], adv_hull
            )

            print(
                f"{item.label:<16} N={budget:<5d} "
                f"uniform={errors[(item.name, 'uniform', budget)]:>10.6f} "
                f"adversarial={errors[(item.name, 'adversarial', budget)]:>10.6f}"
            )

    return results, errors


def summarize_trials(values: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves = {}
    for method, arr in values.items():
        mean = arr.mean(axis=1)
        stderr = arr.std(axis=1, ddof=1) / np.sqrt(TIME_SWEEP_TRIALS)
        ci = 1.96 * stderr
        curves[method] = (mean, np.maximum(mean - ci, np.finfo(float).tiny), mean + ci)
    return curves


def run_time_sweep_ci(
    initial_sets: list[ru.InitialSet],
    estimator: str,
) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run 50-seed time sweep for convex-hull or Christoffel reconstruction."""
    curves: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(RANDOM_SEED + (120_000 if estimator == "christoffel" else 40_000))
    reference_n = CHRISTOFFEL_REFERENCE_SAMPLES if estimator == "christoffel" else REFERENCE_SAMPLES
    reference_initial = {
        item.name: ru.sample_uniform_polygon(rng, item.geom, reference_n)
        for item in initial_sets
    }
    print(f"\nRunning {TIME_SWEEP_TRIALS}-seed {estimator} Hausdorff-vs-time CI sweep:")

    for item in initial_sets:
        reference_clouds = {t: ru.flow(reference_initial[item.name], t) for t in TIME_GRID}
        reference_boundaries = {t: transformed_boundary(item.geom, t) for t in TIME_GRID}
        for budget in SAMPLE_BUDGETS:
            values = {
                "uniform": np.empty((len(TIME_GRID), TIME_SWEEP_TRIALS), dtype=float),
                "adversarial": np.empty((len(TIME_GRID), TIME_SWEEP_TRIALS), dtype=float),
            }
            for t_index, t in enumerate(TIME_GRID):
                for trial in range(TIME_SWEEP_TRIALS):
                    uniform_rng = np.random.default_rng(
                        RANDOM_SEED
                        + ru.stable_seed_offset(item.name, f"uniform-{estimator}", t, budget)
                        + trial
                    )
                    adv_rng = np.random.default_rng(
                        RANDOM_SEED
                        + ru.stable_seed_offset(item.name, f"adv-{estimator}", t, budget)
                        + trial
                    )
                    uniform_points, uniform_hull = ru.run_uniform_sampling(uniform_rng, item.geom, t, budget)
                    adv_points, adv_hull = ru.run_adversarial_sampling(adv_rng, item.geom, t, budget)
                    if estimator == "christoffel":
                        values["uniform"][t_index, trial] = ru.christoffel_support_hausdorff(
                            reference_clouds[t], uniform_points
                        )
                        values["adversarial"][t_index, trial] = ru.christoffel_support_hausdorff(
                            reference_clouds[t], adv_points
                        )
                    else:
                        values["uniform"][t_index, trial] = ru.approximate_boundary_hausdorff(
                            reference_boundaries[t], uniform_hull
                        )
                        values["adversarial"][t_index, trial] = ru.approximate_boundary_hausdorff(
                            reference_boundaries[t], adv_hull
                        )

                print(
                    f"{item.label:<16} t={t:.4f} N={budget:<4d} "
                    f"uniform mean={values['uniform'][t_index].mean():>10.6f} "
                    f"adv mean={values['adversarial'][t_index].mean():>10.6f}"
                )

            for method, curve in summarize_trials(values).items():
                curves[(item.name, method, budget)] = curve

    return curves


def plot_time_sweep_ci(
    curves: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    initial_sets: list[ru.InitialSet],
    figure_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)
    sample_colors = {100: "tab:orange", 1000: "tab:green"}
    method_styles = {
        "uniform": ("-", "o", "uniform"),
        "adversarial": ("--", "s", "adversarial"),
    }

    for ax, item in zip(axes, initial_sets):
        for budget in SAMPLE_BUDGETS:
            color = sample_colors[budget]
            for method, (linestyle, marker, label) in method_styles.items():
                mean, lo, hi = curves[(item.name, method, budget)]
                ax.plot(
                    TIME_GRID,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    lw=2.0,
                    ms=4.6,
                    label=f"N={budget}, {label}",
                )
                ax.fill_between(TIME_GRID, lo, hi, color=color, alpha=0.22)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(TIME_GRID)
        ax.set_xticklabels([f"{t:.4f}" for t in TIME_GRID], rotation=35, ha="right", fontsize=TICK_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.set_title(item.label, fontsize=SUBTITLE_SIZE)
        ax.set_xlabel("time t", fontsize=LABEL_SIZE)
        ax.grid(True, which="both", alpha=0.28)

    axes[0].set_ylabel("Hausdorff distance", fontsize=LABEL_SIZE)
    axes[-1].legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")
    fig.suptitle(title, fontsize=TITLE_SIZE)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    initial_sets = ru.build_equal_area_initial_sets()

    max_initial_x = max(ru.max_x(item.geom) for item in initial_sets)
    T = 0.5 / max_initial_x
    print(f"Chosen T = {T:.8f}; max_x = {max_initial_x:.8f}; T*max_x = {T * max_initial_x:.4f}")
    print("\nInitial-set checks:")
    plot_initial_sets(initial_sets)

    results, errors = fixed_time_comparison(initial_sets, T)
    plot_reachable_clouds(results, initial_sets, T, "uniform", UNIFORM_REACHABLE_FIG)
    plot_reachable_clouds(results, initial_sets, T, "adversarial", ADVERSARIAL_REACHABLE_FIG)
    plot_budget_hausdorff(errors, initial_sets)

    hull_curves = run_time_sweep_ci(initial_sets, "convex-hull")
    plot_time_sweep_ci(hull_curves, initial_sets, HAUSDORFF_TIME_CI_FIG, "Reachable-set approximation error")

    christoffel_curves = run_time_sweep_ci(initial_sets, "christoffel")
    plot_time_sweep_ci(
        christoffel_curves,
        initial_sets,
        HAUSDORFF_TIME_CHRISTOFFEL_CI_FIG,
        "Reachable-set approximation error (Christoffel estimator)",
    )

    print("\nSaved figures:")
    for path in (
        INITIAL_SETS_FIG,
        UNIFORM_REACHABLE_FIG,
        ADVERSARIAL_REACHABLE_FIG,
        HAUSDORFF_FIG,
        HAUSDORFF_TIME_CI_FIG,
        HAUSDORFF_TIME_CHRISTOFFEL_CI_FIG,
    ):
        print(path)


if __name__ == "__main__":
    main()
