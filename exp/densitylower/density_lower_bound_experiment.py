"""Boundary-density ablation on a fixed disk support and linear flow.

Every condition uses the unit disk, ``dx/dt=2x, dy/dt=0``, and ``T=0.5``.
Only the sampling density changes:

    p_beta(x, y) = ((beta + 1)(beta + 2) / (2 pi)) (1-r)^beta,

for beta in {0, 2, 4}.  Samples are exact: R ~ Beta(2, beta+1) and the
angle is uniform.  The script compares convex-hull, packing-ball-union, and
Christoffel estimators and writes raw/aggregate data plus paper figures.

Reproduce the default experiment from the repository root with

    .venv/bin/python -u exp/densitylower/density_lower_bound_experiment.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-reachapprox")

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import shapely
from shapely.geometry import MultiPoint

from exp.quaddynadv.fun.support_estimators import monomial_powers, polynomial_features


@dataclass(frozen=True)
class ExperimentConfig:
    betas: tuple[int, ...] = (0, 2, 4)
    sample_budgets: tuple[int, ...] = (
        10,
        30,
        100,
        300,
        1000,
        3000,
        10_000,
        30_000,
        100_000,
        300_000,
        1_000_000,
    )
    n_seeds: int = 50
    base_seed: int = 20260713
    final_time: float = 0.5
    support_radius: float = 1.0
    support_center: tuple[float, float] = (1.0, 0.0)
    x_growth_rate: float = 2.0
    packing_radius: float = 0.05
    christoffel_degree: int = 6
    christoffel_regularization: float = 1e-6
    grid_resolution: int = 120
    grid_padding_fraction: float = 0.08
    reference_radial_levels: int = 64
    reference_angles: int = 720
    ellipse_boundary_points: int = 4000
    packing_support_angles: int = 4096
    qualitative_budgets: tuple[int, ...] = (1000, 3000, 10_000)
    output_dir: Path = Path(__file__).resolve().parent / "results"


ESTIMATORS = ("convex_hull", "packing", "christoffel")
COLORS = {0: "#2a6fbb", 2: "#e07a1f", 4: "#c83e4d"}
LINESTYLES = {0: "-", 2: "--", 4: "-."}
LABELS = {0: r"$\beta=0$ (uniform)", 2: r"$\beta=2$", 4: r"$\beta=4$"}

RAW_CSV_NAME = "density_boundary_raw.csv"
AGGREGATE_CSV_NAME = "density_boundary_aggregate.csv"
FIGURE7_NAME = "density_boundary.png"
QUALITATIVE_NAME_TEMPLATE = "density_boundary_qualitative_N{budget}.png"
STABILITY_NAME = "density_boundary_resolution_stability.json"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)


def sample_disk_density(
    rng: np.random.Generator,
    n_samples: int,
    beta: int,
    support_radius: float = 1.0,
    support_center: tuple[float, float] = (1.0, 0.0),
) -> np.ndarray:
    """Draw exact samples from p_beta using R ~ Beta(2, beta+1)."""
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    radii = support_radius * rng.beta(2.0, beta + 1.0, size=n_samples)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    points = np.asarray(support_center) + np.column_stack(
        (radii * np.cos(angles), radii * np.sin(angles))
    )
    assert np.all(
        np.linalg.norm(points - np.asarray(support_center), axis=1)
        <= support_radius + 1e-12
    )
    return points


def spatial_density(radius: np.ndarray, beta: int) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    values = ((beta + 1) * (beta + 2) / (2.0 * np.pi)) * np.maximum(1.0 - radius, 0.0) ** beta
    return np.where((radius >= 0.0) & (radius < 1.0), values, 0.0)


def radial_density(radius: np.ndarray, beta: int) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    values = (beta + 1) * (beta + 2) * radius * np.maximum(1.0 - radius, 0.0) ** beta
    return np.where((radius >= 0.0) & (radius <= 1.0), values, 0.0)


def linear_flow(points: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    endpoints = np.asarray(points, dtype=float).copy()
    endpoints[:, 0] *= np.exp(config.x_growth_rate * config.final_time)
    return endpoints


def ellipse_axes(config: ExperimentConfig) -> tuple[float, float]:
    return (
        config.support_radius * np.exp(config.x_growth_rate * config.final_time),
        config.support_radius,
    )


def terminal_center(config: ExperimentConfig) -> np.ndarray:
    return linear_flow(np.asarray(config.support_center, dtype=float)[None, :], config)[0]


def exact_ellipse_boundary(config: ExperimentConfig, n_points: int | None = None) -> np.ndarray:
    count = config.ellipse_boundary_points if n_points is None else n_points
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    a, b = ellipse_axes(config)
    return terminal_center(config) + np.column_stack((a * np.cos(theta), b * np.sin(theta)))


def exact_ellipse_reference(
    config: ExperimentConfig,
    radial_levels: int | None = None,
    angular_levels: int | None = None,
) -> np.ndarray:
    """Deterministic discretization of the analytic filled ellipse."""
    nr = config.reference_radial_levels if radial_levels is None else radial_levels
    nt = config.reference_angles if angular_levels is None else angular_levels
    radii = np.linspace(0.0, 1.0, nr + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    rr, tt = np.meshgrid(radii[1:], theta, indexing="ij")
    a, b = ellipse_axes(config)
    center = terminal_center(config)
    interior = center + np.column_stack(
        (a * rr.ravel() * np.cos(tt).ravel(), b * rr.ravel() * np.sin(tt).ravel())
    )
    return np.vstack((center[None, :], interior, exact_ellipse_boundary(config)))


def make_evaluation_grid(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = ellipse_axes(config)
    center = terminal_center(config)
    pad_x = 2.0 * a * config.grid_padding_fraction
    pad_y = 2.0 * b * config.grid_padding_fraction
    xs = np.linspace(center[0] - a - pad_x, center[0] + a + pad_x, config.grid_resolution)
    ys = np.linspace(center[1] - b - pad_y, center[1] + b + pad_y, config.grid_resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel())), xs, ys


def point_to_filled_ellipse_distance(
    points: np.ndarray,
    axes: tuple[float, float],
    center: np.ndarray,
) -> np.ndarray:
    """Euclidean distance to an axis-aligned filled ellipse via bisection."""
    points = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    a, b = axes
    x = np.abs(points[:, 0])
    y = np.abs(points[:, 1])
    normalized = (x / a) ** 2 + (y / b) ** 2
    distances = np.zeros(points.shape[0], dtype=float)
    outside = normalized > 1.0
    if not np.any(outside):
        return distances

    xo = x[outside]
    yo = y[outside]
    a2 = a * a
    b2 = b * b

    def equation(lam: np.ndarray) -> np.ndarray:
        return (a * xo / (lam + a2)) ** 2 + (b * yo / (lam + b2)) ** 2 - 1.0

    lower = np.zeros_like(xo)
    upper = np.maximum(np.hypot(xo, yo) * max(a, b), 1.0)
    while np.any(equation(upper) > 0.0):
        upper[equation(upper) > 0.0] *= 2.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        positive = equation(middle) > 0.0
        lower[positive] = middle[positive]
        upper[~positive] = middle[~positive]
    lam = 0.5 * (lower + upper)
    projected_x = a2 * xo / (lam + a2)
    projected_y = b2 * yo / (lam + b2)
    distances[outside] = np.hypot(xo - projected_x, yo - projected_y)
    return distances


def fixed_packing_radius(config: ExperimentConfig) -> float:
    """Common ball radius used for every N, beta, and random seed."""
    return config.packing_radius


def greedy_maximal_packing(samples: np.ndarray, bandwidth: float) -> np.ndarray:
    """Greedy maximal h-separated subset using a spatial hash."""
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    buckets: dict[tuple[int, int], list[int]] = {}
    centers: list[np.ndarray] = []
    for sample in np.asarray(samples, dtype=float):
        cell = tuple(np.floor(sample / bandwidth).astype(int))
        accept = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index in buckets.get((cell[0] + dx, cell[1] + dy), ()):
                    if np.linalg.norm(sample - centers[index]) <= bandwidth:
                        accept = False
                        break
                if not accept:
                    break
            if not accept:
                break
        if accept:
            index = len(centers)
            centers.append(sample.copy())
            buckets.setdefault(cell, []).append(index)
    selected = np.asarray(centers, dtype=float)
    verify_maximal_packing(samples, selected, bandwidth)
    return selected


def verify_maximal_packing(samples: np.ndarray, centers: np.ndarray, bandwidth: float) -> None:
    if centers.shape[0] > 1:
        nearest_pair = cKDTree(centers).query(centers, k=2)[0][:, 1].min()
        assert nearest_pair > bandwidth - 1e-12
    coverage = cKDTree(centers).query(samples, k=1)[0].max()
    assert coverage <= bandwidth + 1e-12


def packing_outer_error(centers: np.ndarray, bandwidth: float, config: ExperimentConfig) -> float:
    """Directed union-of-balls-to-ellipse distance using support functions."""
    if centers.shape[0] >= 3:
        try:
            hull = ConvexHull(centers)
            support_centers = centers[hull.vertices]
        except Exception:
            support_centers = centers
    else:
        support_centers = centers
    theta = np.linspace(0.0, 2.0 * np.pi, config.packing_support_angles, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    center_support = np.full(directions.shape[0], -np.inf)
    for start in range(0, support_centers.shape[0], 1024):
        center_support = np.maximum(
            center_support,
            np.max(support_centers[start : start + 1024] @ directions.T, axis=0),
        )
    a, b = ellipse_axes(config)
    ellipse_support = (
        terminal_center(config) @ directions.T
        + np.sqrt((a * directions[:, 0]) ** 2 + (b * directions[:, 1]) ** 2)
    )
    return float(max(0.0, np.max(center_support + bandwidth - ellipse_support)))


def christoffel_mask(
    samples: np.ndarray,
    grid_points: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, float]:
    """Fixed Christoffel estimator using stable linear solves."""
    lower = grid_points.min(axis=0)
    upper = grid_points.max(axis=0)
    center = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    sample_scaled = (samples - center) / scale
    grid_scaled = (grid_points - center) / scale
    powers = monomial_powers(config.christoffel_degree)
    phi_samples = polynomial_features(sample_scaled, powers)
    gram = (phi_samples.T @ phi_samples) / samples.shape[0]
    ridge = config.christoffel_regularization * max(float(np.trace(gram)) / gram.shape[0], 1.0)
    matrix = gram + ridge * np.eye(gram.shape[0])
    solved_samples = np.linalg.solve(matrix, phi_samples.T).T
    sample_values = np.einsum("ij,ij->i", phi_samples, solved_samples)
    threshold = float(np.max(sample_values)) * (1.0 + 1e-10)
    phi_grid = polynomial_features(grid_scaled, powers)
    solved_grid = np.linalg.solve(matrix, phi_grid.T).T
    grid_values = np.einsum("ij,ij->i", phi_grid, solved_grid)
    mask = grid_values <= threshold
    if not np.any(mask):
        nearest = cKDTree(grid_points).query(samples[:1], k=1)[1][0]
        mask[int(nearest)] = True
    return mask, threshold


def directed_errors(
    estimator: str,
    samples: np.ndarray,
    reference_points: np.ndarray,
    grid_points: np.ndarray,
    config: ExperimentConfig,
) -> tuple[float, float, int, float, float]:
    """Return inner, outer, center count, bandwidth, and threshold."""
    if estimator == "convex_hull":
        hull = MultiPoint(samples).convex_hull
        ref_geometries = shapely.points(reference_points[:, 0], reference_points[:, 1])
        inner = float(np.max(shapely.distance(ref_geometries, hull)))
        return inner, 0.0, 0, float("nan"), float("nan")

    if estimator == "packing":
        bandwidth = fixed_packing_radius(config)
        centers = greedy_maximal_packing(samples, bandwidth)
        nearest = cKDTree(centers).query(reference_points, k=1)[0]
        inner = float(np.max(np.maximum(nearest - bandwidth, 0.0)))
        outer = packing_outer_error(centers, bandwidth, config)
        return inner, outer, int(centers.shape[0]), bandwidth, float("nan")

    if estimator == "christoffel":
        mask, threshold = christoffel_mask(samples, grid_points, config)
        estimate_points = np.vstack((grid_points[mask], samples))
        inner = float(cKDTree(estimate_points).query(reference_points, k=1)[0].max())
        outer = float(
            point_to_filled_ellipse_distance(
                estimate_points,
                ellipse_axes(config),
                terminal_center(config),
            ).max()
        )
        return inner, outer, 0, float("nan"), threshold

    raise ValueError(f"unknown estimator: {estimator}")


def seed_for(config: ExperimentConfig, budget_index: int, seed_index: int) -> int:
    return config.base_seed + 10_000 * budget_index + seed_index


def run_experiment(config: ExperimentConfig) -> list[dict[str, object]]:
    reference = exact_ellipse_reference(config)
    grid, _, _ = make_evaluation_grid(config)
    rows: list[dict[str, object]] = []
    for budget_index, n_samples in enumerate(config.sample_budgets):
        print(f"N={n_samples}")
        for beta in config.betas:
            beta_errors: list[float] = []
            for seed_index in range(config.n_seeds):
                seed = seed_for(config, budget_index, seed_index)
                initial = sample_disk_density(
                    np.random.default_rng(seed),
                    n_samples,
                    beta,
                    config.support_radius,
                    config.support_center,
                )
                endpoints = linear_flow(initial, config)
                for estimator in ESTIMATORS:
                    inner, outer, centers, bandwidth, threshold = directed_errors(
                        estimator, endpoints, reference, grid, config
                    )
                    symmetric = max(inner, outer)
                    beta_errors.append(symmetric) if estimator == "christoffel" else None
                    rows.append(
                        {
                            "beta": beta,
                            "N": n_samples,
                            "seed_index": seed_index,
                            "seed": seed,
                            "estimator": estimator,
                            "hausdorff": symmetric,
                            "inner_directed": inner,
                            "outer_directed": outer,
                            "packing_centers": centers,
                            "packing_radius": bandwidth,
                            "christoffel_degree": config.christoffel_degree,
                            "christoffel_regularization": config.christoffel_regularization,
                            "christoffel_threshold": threshold,
                            "grid_resolution": config.grid_resolution,
                        }
                    )
            print(f"  beta={beta}: Christoffel mean={np.mean(beta_errors):.5f}")
    assert_complete_results(rows, config)
    return rows


def assert_complete_results(rows: list[dict[str, object]], config: ExperimentConfig) -> None:
    expected = {
        (beta, n, seed_index, estimator)
        for beta in config.betas
        for n in config.sample_budgets
        for seed_index in range(config.n_seeds)
        for estimator in ESTIMATORS
    }
    actual = {(int(r["beta"]), int(r["N"]), int(r["seed_index"]), str(r["estimator"])) for r in rows}
    if actual != expected:
        raise AssertionError(f"missing={len(expected-actual)}, unexpected={len(actual-expected)}")


RAW_FIELDS = (
    "beta", "N", "seed_index", "seed", "estimator", "hausdorff",
    "inner_directed", "outer_directed", "packing_centers", "packing_radius",
    "christoffel_degree", "christoffel_regularization",
    "christoffel_threshold", "grid_resolution",
)


def write_raw_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    keys = sorted({(int(r["beta"]), int(r["N"]), str(r["estimator"])) for r in rows})
    for beta, n_samples, estimator in keys:
        selected = [r for r in rows if int(r["beta"]) == beta and int(r["N"]) == n_samples and r["estimator"] == estimator]
        row: dict[str, object] = {"beta": beta, "N": n_samples, "estimator": estimator, "n_seeds": len(selected)}
        for metric in ("hausdorff", "inner_directed", "outer_directed", "packing_centers"):
            values = np.asarray([float(r[metric]) for r in selected])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        output.append(row)
    return output


def write_dict_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def packing_radius_tag(radius: float) -> str:
    return f"{radius:g}".replace(".", "p")


def write_packing_summary_csv(
    aggregate: list[dict[str, object]],
    path: Path,
    radius: float,
) -> None:
    rows = [
        {"packing_radius": radius, **row}
        for row in aggregate
        if row["estimator"] == "packing"
    ]
    write_dict_csv(rows, path)


def aggregate_lookup(aggregate: list[dict[str, object]]) -> dict[tuple[int, int, str], dict[str, object]]:
    return {(int(r["beta"]), int(r["N"]), str(r["estimator"])): r for r in aggregate}


def plot_error_panel(
    ax: plt.Axes,
    estimator: str,
    title: str,
    aggregate: list[dict[str, object]],
    config: ExperimentConfig,
) -> None:
    lookup = aggregate_lookup(aggregate)
    budgets = np.asarray(config.sample_budgets)
    for beta in config.betas:
        selected = [lookup[(beta, n, estimator)] for n in config.sample_budgets]
        mean = np.asarray([r["hausdorff_mean"] for r in selected], dtype=float)
        q05 = np.asarray([r["hausdorff_q05"] for r in selected], dtype=float)
        q95 = np.asarray([r["hausdorff_q95"] for r in selected], dtype=float)
        ax.plot(budgets, mean, color=COLORS[beta], linestyle=LINESTYLES[beta], marker="o", ms=4, lw=1.8, label=LABELS[beta])
        ax.fill_between(budgets, q05, q95, color=COLORS[beta], alpha=0.14, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"Sample size $N$")
    ax.set_ylabel(r"Hausdorff error $d_H$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)


def plot_figure7(aggregate: list[dict[str, object]], config: ExperimentConfig) -> Path:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 4.4),
        constrained_layout=True,
        sharey=True,
    )
    plot_error_panel(axes[0], "convex_hull", "(a) Convex hull", aggregate, config)
    plot_error_panel(axes[1], "packing", "(b) Packing-ball union", aggregate, config)
    plot_error_panel(axes[2], "christoffel", "(c) Christoffel function", aggregate, config)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, fontsize=10, loc="upper right")
    for ax in axes[1:]:
        ax.set_ylabel("")
    path = config.output_dir / FIGURE7_NAME
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_qualitative(config: ExperimentConfig, qualitative_budget: int) -> Path:
    if qualitative_budget not in config.sample_budgets:
        raise ValueError("qualitative budget must be one of the sample budgets")
    budget_index = config.sample_budgets.index(qualitative_budget)
    seed = seed_for(config, budget_index, 0)
    grid, xs, ys = make_evaluation_grid(config)
    boundary = exact_ellipse_boundary(config)
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 5.8), constrained_layout=True, sharex=True, sharey=True)
    for row, beta in enumerate(config.betas):
        initial = sample_disk_density(
            np.random.default_rng(seed),
            qualitative_budget,
            beta,
            config.support_radius,
            config.support_center,
        )
        samples = linear_flow(initial, config)
        for column, estimator in enumerate(ESTIMATORS):
            ax = axes[row, column]
            ax.plot(boundary[:, 0], boundary[:, 1], color="black", lw=1.4, label="exact ellipse")
            ax.scatter(samples[:, 0], samples[:, 1], s=5, color="#777777", alpha=0.45, linewidths=0)
            if estimator == "convex_hull":
                hull = MultiPoint(samples).convex_hull
                if hull.geom_type == "Polygon":
                    coords = np.asarray(hull.exterior.coords)
                    ax.fill(coords[:, 0], coords[:, 1], color=COLORS[beta], alpha=0.18)
                    ax.plot(coords[:, 0], coords[:, 1], color=COLORS[beta], lw=1.4)
            elif estimator == "packing":
                h = fixed_packing_radius(config)
                centers = greedy_maximal_packing(samples, h)
                patches = [Circle(center, h) for center in centers]
                collection = PatchCollection(patches, facecolor=COLORS[beta], edgecolor=COLORS[beta], alpha=0.12, linewidth=0.35)
                ax.add_collection(collection)
            else:
                mask, _ = christoffel_mask(samples, grid, config)
                zz = mask.reshape(len(ys), len(xs)).astype(float)
                ax.contourf(xs, ys, zz, levels=[0.5, 1.5], colors=[COLORS[beta]], alpha=0.18)
                ax.contour(xs, ys, zz, levels=[0.5], colors=[COLORS[beta]], linewidths=1.2)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.15)
            if row == 0:
                ax.set_title(("Convex hull", "Packing balls", "Christoffel")[column])
            if column == 0:
                ax.set_ylabel(LABELS[beta] + "\n$Y$")
            if row == 2:
                ax.set_xlabel("$X$")
    fig.suptitle(rf"Qualitative estimates at $N={qualitative_budget}$", fontsize=17)
    path = config.output_dir / QUALITATIVE_NAME_TEMPLATE.format(
        budget=qualitative_budget
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def validate_resolution_stability(config: ExperimentConfig) -> Path:
    beta = max(config.betas)
    n_samples = max(config.sample_budgets)
    budget_index = config.sample_budgets.index(n_samples)
    seed = seed_for(config, budget_index, 0)
    initial = sample_disk_density(
        np.random.default_rng(seed),
        n_samples,
        beta,
        config.support_radius,
        config.support_center,
    )
    endpoints = linear_flow(initial, config)
    grid, _, _ = make_evaluation_grid(config)
    fine_grid_config = replace(config, grid_resolution=int(np.ceil(1.5 * config.grid_resolution)))
    fine_grid, _, _ = make_evaluation_grid(fine_grid_config)
    coarse = exact_ellipse_reference(config)
    fine = exact_ellipse_reference(
        config,
        radial_levels=int(np.ceil(1.5 * config.reference_radial_levels)),
        angular_levels=int(np.ceil(1.5 * config.reference_angles)),
    )
    report: dict[str, object] = {
        "beta": beta,
        "N": n_samples,
        "seed": seed,
        "coarse_reference_points": int(coarse.shape[0]),
        "fine_reference_points": int(fine.shape[0]),
        "coarse_grid_resolution": config.grid_resolution,
        "fine_grid_resolution": fine_grid_config.grid_resolution,
        "tolerance": 0.03,
        "estimators": {},
    }
    for estimator in ESTIMATORS:
        coarse_values = directed_errors(estimator, endpoints, coarse, grid, config)
        if estimator == "christoffel":
            fine_values = directed_errors(estimator, endpoints, fine, fine_grid, fine_grid_config)
        else:
            fine_values = directed_errors(estimator, endpoints, fine, grid, config)
        coarse_h = max(coarse_values[0], coarse_values[1])
        fine_h = max(fine_values[0], fine_values[1])
        difference = abs(coarse_h - fine_h)
        if difference > 0.03:
            raise AssertionError(f"{estimator} resolution difference {difference:.5f} exceeds tolerance")
        report["estimators"][estimator] = {"coarse": coarse_h, "fine": fine_h, "absolute_difference": difference}
    path = config.output_dir / STABILITY_NAME
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", default="0,2,4")
    parser.add_argument(
        "--budgets",
        default="10,30,100,300,1000,3000,10000,30000,100000,300000,1000000",
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--packing-radius", type=float, default=0.05)
    parser.add_argument("--christoffel-degree", type=int, default=6)
    parser.add_argument("--christoffel-regularization", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig().output_dir)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        betas=parse_int_tuple(args.betas),
        sample_budgets=parse_int_tuple(args.budgets),
        n_seeds=args.seeds,
        packing_radius=args.packing_radius,
        christoffel_degree=args.christoffel_degree,
        christoffel_regularization=args.christoffel_regularization,
        output_dir=args.output_dir.resolve(),
    )


def main(argv: Iterable[str] | None = None) -> None:
    config = config_from_args(build_parser().parse_args(argv))
    if config.n_seeds < 1 or config.packing_radius <= 0.0:
        raise ValueError("seeds and packing radius must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_experiment(config)
    aggregate = aggregate_results(rows)
    raw_path = config.output_dir / RAW_CSV_NAME
    aggregate_path = config.output_dir / AGGREGATE_CSV_NAME
    packing_path = config.output_dir / (
        f"packing_radius_{packing_radius_tag(config.packing_radius)}.csv"
    )
    write_raw_csv(rows, raw_path)
    write_dict_csv(aggregate, aggregate_path)
    write_packing_summary_csv(aggregate, packing_path, config.packing_radius)
    qualitative_paths = tuple(
        plot_qualitative(config, budget) for budget in config.qualitative_budgets
    )
    generated = (
        plot_figure7(aggregate, config),
        *qualitative_paths,
        validate_resolution_stability(config),
        raw_path,
        aggregate_path,
        packing_path,
    )
    for path in generated:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
