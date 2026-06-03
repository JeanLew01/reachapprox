"""Uniform/adversarial sampling experiments for dx/dt=x^2, dy/dt=0.

Run from /home/jixia/exp with:

    .venv/bin/python -u reachapprox/exp/quaddynadv/quaddyn_adv_experiment.py

Shared geometry, sampling, flow, support-estimator, and Hausdorff utilities live
in :mod:`reachapprox.exp.quaddynadv.fun`; this file only defines the experiment
workflow and figures for the quadratic non-Lipschitz dynamics example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from reachapprox.exp.quaddynadv import fun as ru


SAMPLE_BUDGETS = (100, 1000)
TIME_GRID = tuple(float(t) for t in np.linspace(0.01, 0.29, 17))
LINEAR_TIME_GRID = tuple(float(t) for t in np.linspace(0.01, 2.0, 17))
Y_SCALE_TIME_MAX = 0.28
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
LEGEND_SIZE = 15
INSET_TICK_SIZE = 12
INSET_TITLE_SIZE = 13
TIME_SWEEP_BOUNDARY_POINTS = 700

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)


@dataclass(frozen=True)
class DynamicsSpec:
    name: str
    label: str
    flow: Callable[[np.ndarray, float], np.ndarray]
    jacobian: Callable[[np.ndarray, float], np.ndarray]


def linear_flow(points: np.ndarray, T: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    out = points.copy()
    out[:, 0] = np.exp(T) * out[:, 0]
    return out


def linear_flow_jacobian(points: np.ndarray, T: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    jac = np.zeros((points.shape[0], 2, 2), dtype=float)
    jac[:, 0, 0] = np.exp(T)
    jac[:, 1, 1] = 1.0
    return jac


DYNAMICS_SPECS = (
    DynamicsSpec("quadratic", r"$\dot{x}=x^2,\ \dot{y}=0$", ru.flow, ru.flow_jacobian),
    DynamicsSpec("linear", r"$\dot{x}=x,\ \dot{y}=0$", linear_flow, linear_flow_jacobian),
)


def main_time_grid(dynamics: DynamicsSpec) -> tuple[float, ...]:
    return LINEAR_TIME_GRID if dynamics.name == "linear" else TIME_GRID


def sweep_time_grids(dynamics: DynamicsSpec, item: ru.InitialSet) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if dynamics.name == "linear":
        return (("main", LINEAR_TIME_GRID), ("inset", TIME_GRID))
    return (("main", TIME_GRID),)


def ordered_initial_sets() -> list[ru.InitialSet]:
    """Return sets in requested plotting order: circle, opened triangle, triangle."""
    by_name = {item.name: item for item in ru.build_equal_area_initial_sets()}
    disk = by_name["disk"]
    return [
        ru.InitialSet(disk.name, "Circle", disk.geom, disk.color),
        by_name["opened"],
        by_name["triangle"],
    ]


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


def transformed_boundary_for_dynamics(
    dynamics: DynamicsSpec,
    geom,
    T: float,
    n_points: int = 1_200,
) -> np.ndarray:
    return dynamics.flow(ru.boundary_points(geom, n_points), T)


def adversarial_gradient_for_dynamics(
    dynamics: DynamicsSpec,
    points: np.ndarray,
    T: float,
    c: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    endpoints = dynamics.flow(points, T)
    diff = endpoints - c
    qdiff = diff @ Q.T
    jac = dynamics.jacobian(points, T)
    return 2.0 * np.einsum("nij,nj->ni", np.swapaxes(jac, 1, 2), qdiff)


def run_uniform_sampling_for_dynamics(
    rng: np.random.Generator,
    geom,
    dynamics: DynamicsSpec,
    T: float,
    n_samples: int,
) -> tuple[np.ndarray, object]:
    x0 = ru.sample_uniform_polygon(rng, geom, n_samples)
    endpoints = dynamics.flow(x0, T)
    return endpoints, ru.convex_hull_polygon(endpoints)


def run_adversarial_sampling_for_dynamics(
    rng: np.random.Generator,
    geom,
    dynamics: DynamicsSpec,
    T: float,
    total_budget: int,
    n_adv: int = 9,
    eta: float = 0.018,
) -> tuple[np.ndarray, object]:
    if total_budget % (n_adv + 1) != 0:
        raise ValueError("total_budget must be divisible by n_adv + 1")

    m_particles = total_budget // (n_adv + 1)
    particles = ru.sample_uniform_polygon(rng, geom, m_particles)
    endpoints = [dynamics.flow(particles, T)]

    for _ in range(n_adv):
        accumulated = np.vstack(endpoints)
        c, Q = ru.endpoint_center_and_Q(accumulated)
        grad = adversarial_gradient_for_dynamics(dynamics, particles, T, c, Q)
        particles = ru.project_points_to_set(particles + eta * grad, geom)
        endpoints.append(dynamics.flow(particles, T))

    endpoint_cloud = np.vstack(endpoints)
    return endpoint_cloud, ru.convex_hull_polygon(endpoint_cloud)


def approximate_boundary_hausdorff_fast(reference_boundary: np.ndarray, estimate_hull) -> float:
    estimate_boundary = ru.boundary_points(estimate_hull, TIME_SWEEP_BOUNDARY_POINTS)
    tree_ref = cKDTree(reference_boundary)
    tree_est = cKDTree(estimate_boundary)
    ref_to_est = tree_est.query(reference_boundary, k=1)[0].max()
    est_to_ref = tree_ref.query(estimate_boundary, k=1)[0].max()
    return float(max(ref_to_est, est_to_ref))


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
        valid_counts = np.sum(np.isfinite(arr), axis=1)
        mean = np.full(arr.shape[0], np.nan, dtype=float)
        stderr = np.full(arr.shape[0], np.nan, dtype=float)
        valid_rows = valid_counts > 0
        mean[valid_rows] = np.nanmean(arr[valid_rows], axis=1)
        multi_sample_rows = valid_counts > 1
        stderr[multi_sample_rows] = (
            np.nanstd(arr[multi_sample_rows], axis=1, ddof=1) / np.sqrt(valid_counts[multi_sample_rows])
        )
        stderr[valid_counts == 1] = 0.0
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
                        values["uniform"][t_index, trial] = approximate_boundary_hausdorff_fast(
                            reference_boundaries[t], uniform_hull
                        )
                        values["adversarial"][t_index, trial] = approximate_boundary_hausdorff_fast(
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


def run_multi_dynamics_time_sweep_ci(
    initial_sets: list[ru.InitialSet],
    dynamics_specs: tuple[DynamicsSpec, ...] = DYNAMICS_SPECS,
) -> dict[tuple[str, str, str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run the requested two-dynamics, three-initial-set Hausdorff-vs-time sweep."""
    curves: dict[tuple[str, str, str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(RANDOM_SEED + 240_000)
    reference_initial = {
        item.name: ru.sample_uniform_polygon(rng, item.geom, REFERENCE_SAMPLES)
        for item in initial_sets
    }
    print(f"\nRunning {TIME_SWEEP_TRIALS}-seed two-dynamics convex-hull Hausdorff-vs-time CI sweep:")

    for dynamics in dynamics_specs:
        print(f"\nDynamics: {dynamics.label}")
        for item in initial_sets:
            for grid_name, time_grid in sweep_time_grids(dynamics, item):
                for budget in SAMPLE_BUDGETS:
                    values = {
                        "uniform": np.empty((len(time_grid), TIME_SWEEP_TRIALS), dtype=float),
                        "adversarial": np.empty((len(time_grid), TIME_SWEEP_TRIALS), dtype=float),
                    }
                    for t_index, t in enumerate(time_grid):
                        try:
                            reference_boundary = transformed_boundary_for_dynamics(dynamics, item.geom, t)
                        except ValueError:
                            values["uniform"][t_index, :] = np.nan
                            values["adversarial"][t_index, :] = np.nan
                            print(
                                f"{dynamics.name:<10} {grid_name:<5} {item.label:<16} t={t:.4f} N={budget:<4d} "
                                "singular; omitted from plot"
                            )
                            continue
                        for trial in range(TIME_SWEEP_TRIALS):
                            uniform_rng = np.random.default_rng(
                                RANDOM_SEED
                                + ru.stable_seed_offset(
                                    item.name, f"{dynamics.name}-{grid_name}-uniform-hull", t, budget
                                )
                                + trial
                            )
                            adv_rng = np.random.default_rng(
                                RANDOM_SEED
                                + ru.stable_seed_offset(
                                    item.name, f"{dynamics.name}-{grid_name}-adv-hull", t, budget
                                )
                                + trial
                            )
                            _, uniform_hull = run_uniform_sampling_for_dynamics(
                                uniform_rng, item.geom, dynamics, t, budget
                            )
                            _, adv_hull = run_adversarial_sampling_for_dynamics(
                                adv_rng, item.geom, dynamics, t, budget
                            )
                            values["uniform"][t_index, trial] = approximate_boundary_hausdorff_fast(
                                reference_boundary, uniform_hull
                            )
                            values["adversarial"][t_index, trial] = approximate_boundary_hausdorff_fast(
                                reference_boundary, adv_hull
                            )

                        print(
                            f"{dynamics.name:<10} {grid_name:<5} {item.label:<16} t={t:.4f} N={budget:<4d} "
                            f"uniform mean={values['uniform'][t_index].mean():>10.6f} "
                            f"adv mean={values['adversarial'][t_index].mean():>10.6f}"
                        )

                    for method, curve in summarize_trials(values).items():
                        curves[(dynamics.name, item.name, method, budget, grid_name)] = curve

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


def plot_multi_dynamics_time_sweep_ci(
    curves: dict[tuple[str, str, str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    initial_sets: list[ru.InitialSet],
    figure_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(DYNAMICS_SPECS),
        len(initial_sets),
        figsize=(16.4, 9.4),
        constrained_layout=True,
        sharey=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.03, wspace=0.015, hspace=0.04)
    sample_colors = {100: "tab:orange", 1000: "tab:green"}
    method_styles = {
        "uniform": ("-", "o", "uniform"),
        "adversarial": ("--", "s", "adversarial"),
    }

    def add_initial_set_inset(ax, item: ru.InitialSet) -> None:
        inset = ax.inset_axes([0.61, 0.68, 0.27, 0.31])
        inset.add_patch(
            ru.polygon_patch(
                item.geom,
                facecolor=item.color,
                alpha=0.30,
                edgecolor=item.color,
                lw=1.6,
            )
        )
        centroid = item.geom.centroid
        inset.scatter([centroid.x], [centroid.y], color="black", s=7, zorder=4)
        minx, miny, maxx, maxy = item.geom.bounds
        pad = 0.12 * max(maxx - minx, maxy - miny)
        inset.set_xlim(minx - pad, maxx + pad)
        inset.set_ylim(miny - pad, maxy + pad)
        inset.set_aspect("equal", adjustable="box")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.patch.set_alpha(0.0)
        for spine in inset.spines.values():
            spine.set_visible(False)

    y_lower = np.inf
    y_upper = 0.0
    for dynamics in DYNAMICS_SPECS:
        for item in initial_sets:
            item_main_time_grid = dict(sweep_time_grids(dynamics, item))["main"]
            for budget in SAMPLE_BUDGETS:
                for method in method_styles:
                    mean, lo, hi = curves[(dynamics.name, item.name, method, budget, "main")]
                    time_mask = np.asarray(item_main_time_grid) <= Y_SCALE_TIME_MAX
                    if dynamics.name == "linear":
                        time_mask = np.ones_like(time_mask, dtype=bool)
                    y_lower = min(y_lower, float(np.min(lo[time_mask])), float(np.min(mean[time_mask])))
                    y_upper = max(y_upper, float(np.max(hi[time_mask])), float(np.max(mean[time_mask])))
                    if dynamics.name == "linear":
                        inset_mean, inset_lo, inset_hi = curves[
                            (dynamics.name, item.name, method, budget, "inset")
                        ]
                        inset_time_mask = np.asarray(TIME_GRID) <= Y_SCALE_TIME_MAX
                        y_lower = min(
                            y_lower,
                            float(np.min(inset_lo[inset_time_mask])),
                            float(np.min(inset_mean[inset_time_mask])),
                        )
                        y_upper = max(
                            y_upper,
                            float(np.max(inset_hi[inset_time_mask])),
                            float(np.max(inset_mean[inset_time_mask])),
                        )

    y_lower = max(y_lower * 0.75, np.finfo(float).tiny)
    y_upper = y_upper * 1.35

    legend_handles = []
    legend_labels = []
    for row, dynamics in enumerate(DYNAMICS_SPECS):
        for col, item in enumerate(initial_sets):
            time_grid = dict(sweep_time_grids(dynamics, item))["main"]
            tick_time_grid = main_time_grid(dynamics)
            ax = axes[row, col]
            for budget in SAMPLE_BUDGETS:
                color = sample_colors[budget]
                for method, (linestyle, marker, label) in method_styles.items():
                    mean, lo, hi = curves[(dynamics.name, item.name, method, budget, "main")]
                    (line,) = ax.plot(
                        time_grid,
                        mean,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        lw=2.0,
                        ms=4.6,
                        label=f"N={budget}, {label}",
                    )
                    ax.fill_between(time_grid, lo, hi, color=color, alpha=0.22)
                    if row == 0 and col == 0:
                        legend_handles.append(line)
                        legend_labels.append(f"N={budget}, {label}")

            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.set_xlim(min(time_grid), max(time_grid))
            ax.set_ylim(y_lower, y_upper)
            ax.set_xticks(tick_time_grid)
            ax.set_xticklabels([f"{t:.4f}" for t in tick_time_grid], rotation=35, ha="right", fontsize=TICK_SIZE)
            ax.grid(True, which="both", alpha=0.28)
            ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
            ax.tick_params(axis="y", which="both", labelleft=(col == 0))

            if row == 0:
                ax.set_title(item.label, fontsize=SUBTITLE_SIZE)
            if col == 0:
                ax.set_ylabel(f"{dynamics.label}\nHausdorff distance", fontsize=LABEL_SIZE)
            if row == len(DYNAMICS_SPECS) - 1:
                ax.set_xlabel("time t", fontsize=LABEL_SIZE)

            if dynamics.name == "quadratic":
                add_initial_set_inset(ax, item)

            if dynamics.name == "linear":
                inset = ax.inset_axes([0.12, 0.56, 0.43, 0.37])
                for budget in SAMPLE_BUDGETS:
                    color = sample_colors[budget]
                    for method, (linestyle, marker, label) in method_styles.items():
                        mean, lo, hi = curves[(dynamics.name, item.name, method, budget, "inset")]
                        inset.plot(
                            TIME_GRID,
                            mean,
                            color=color,
                            linestyle=linestyle,
                            marker=marker,
                            lw=1.2,
                            ms=2.4,
                        )
                        inset.fill_between(TIME_GRID, lo, hi, color=color, alpha=0.18)
                inset.set_xscale("linear")
                inset.set_yscale("log")
                inset.set_xlim(min(TIME_GRID), max(TIME_GRID))
                inset.set_xticks((TIME_GRID[0], TIME_GRID[4], TIME_GRID[8], TIME_GRID[-1]))
                inset.set_xticklabels(
                    [f"{t:.4f}" for t in (TIME_GRID[0], TIME_GRID[4], TIME_GRID[8], TIME_GRID[-1])],
                    fontsize=INSET_TICK_SIZE,
                    rotation=35,
                    ha="right",
                )
                inset.tick_params(axis="y", which="both", labelsize=INSET_TICK_SIZE)
                inset.tick_params(axis="both", which="both", length=2.0)
                inset.grid(True, which="both", alpha=0.22)
                inset.set_title(r"$t\in[0.01,0.29]$", fontsize=INSET_TITLE_SIZE)

    axes[0, 0].legend(
        legend_handles,
        legend_labels,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="upper left",
    )
    fig.suptitle("Reachable-set approximation error", fontsize=TITLE_SIZE)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    initial_sets = ordered_initial_sets()

    max_initial_x = max(ru.max_x(item.geom) for item in initial_sets)
    T = 0.5 / max_initial_x
    print(f"Chosen T = {T:.8f}; max_x = {max_initial_x:.8f}; T*max_x = {T * max_initial_x:.4f}")
    print("\nInitial-set checks:")
    plot_initial_sets(initial_sets)

    hull_curves = run_multi_dynamics_time_sweep_ci(initial_sets)
    plot_multi_dynamics_time_sweep_ci(hull_curves, initial_sets, HAUSDORFF_TIME_CI_FIG)

    print("\nSaved figures:")
    for path in (
        INITIAL_SETS_FIG,
        HAUSDORFF_TIME_CI_FIG,
    ):
        print(path)


if __name__ == "__main__":
    main()
