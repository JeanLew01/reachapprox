"""Dimension-dependent approximation of robot-arm uncertainty propagation.

Run from /home/jixia/exp with:

    .venv/bin/python -u reachapprox/exp/robotarm/robotarm_dim_scaling.py

The reported metric is a point-cloud directed Hausdorff Distance from a
reference terminal endpoint cloud to the sampled terminal endpoint cloud. This
matches the covering-law interpretation N = C r^{-d}.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reachapprox.exp.robotarm.fun.mujoco_n_link_arm import (
    DEFAULT_GAMMA,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_LAMBDA,
    DEFAULT_SIGMA,
    DEFAULT_TAU_LIMIT,
)
from reachapprox.exp.robotarm.fun.dim_scaling import (
    DEFAULT_COVERAGE_SUBSET,
    DEFAULT_LINK_COUNTS,
    DEFAULT_N_REF,
    DEFAULT_N_SEEDS,
    DEFAULT_SAMPLE_BUDGETS,
    EXPERIMENT_SEED,
    METRIC_IMPLEMENTATION,
    METRIC_NAME,
    REFERENCE_SEED,
    RHO_Q,
    RHO_V,
    T_HORIZON,
    controller_label,
    directed_hausdorff_to_convex_hull,
    mean_and_ci,
    parse_int_tuple,
    propagate_endpoints,
    sample_initial_box,
)

FIG_DIR = Path("CoRL_2026/fig")
CSV_PATH = FIG_DIR / "robotarm_dim_scaling_results.csv"
UNIFORM_FIG = FIG_DIR / "robotarm_uniform_dim_scaling_hausdorff_vs_N.png"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)

TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 13
LEGEND_SIZE = 12


@dataclass(frozen=True)
class ExperimentResult:
    method: str
    n: int
    state_dim: int
    budget: int
    seed: int
    error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_values", default="2,3,4", help="Comma-separated link counts, e.g. 2,3,4.")
    parser.add_argument("--budgets", default="1,3,10,30,100,300,1000,3000")
    parser.add_argument("--n_seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--N_ref", type=int, default=DEFAULT_N_REF)
    parser.add_argument("--coverage_subset", type=int, default=DEFAULT_COVERAGE_SUBSET)
    parser.add_argument("--T", type=float, default=T_HORIZON)
    parser.add_argument("--rho_q", type=float, default=RHO_Q)
    parser.add_argument("--rho_v", type=float, default=RHO_V)
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    parser.add_argument("--lambda_gain", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--tau_limit", type=float, default=DEFAULT_TAU_LIMIT)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--fig", type=Path, default=UNIFORM_FIG)
    parser.add_argument("--force", action="store_true", help="Recompute even if a matching CSV exists.")
    return parser.parse_args()




def run_experiment(
    n_values: tuple[int, ...],
    budgets: tuple[int, ...],
    n_seeds: int,
    N_ref: int,
    coverage_subset: int,
    T: float,
    rho_q: float,
    rho_v: float,
    kp: float,
    kd: float,
    lambda_gain: float,
    gamma: float,
    sigma: float,
    tau_limit: float,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    for n in n_values:
        state_dim = 2 * n
        ref_rng = np.random.default_rng(REFERENCE_SEED + n)
        X_ref = sample_initial_box(ref_rng, N_ref, n, rho_q, rho_v)
        Y_ref = propagate_endpoints(X_ref, n, T, kp, kd, lambda_gain, gamma, sigma, tau_limit)
        subset_size = min(coverage_subset, N_ref)
        print(f"\nRobot arm n={n}, state_dim={state_dim}: reference {N_ref}, subset {subset_size}")

        for budget in budgets:
            for seed_index in range(n_seeds):
                seed = EXPERIMENT_SEED + 10_000 * n + 100 * seed_index + budget
                rng = np.random.default_rng(seed)
                subset_idx = rng.choice(N_ref, size=subset_size, replace=False)
                Y_ref_subset = Y_ref[subset_idx]
                X_uniform = sample_initial_box(rng, budget, n, rho_q, rho_v)
                Y_uniform = propagate_endpoints(X_uniform, n, T, kp, kd, lambda_gain, gamma, sigma, tau_limit)
                metric_rng = np.random.default_rng(seed + 9_000_000)
                error = directed_hausdorff_to_convex_hull(Y_ref_subset, Y_uniform, metric_rng)
                results.append(ExperimentResult("uniform", n, state_dim, budget, seed, error))

            print(f"  finished N={budget}")

    return results


def save_results_csv(
    results: list[ExperimentResult],
    path: Path,
    T: float,
    rho_q: float,
    rho_v: float,
    controller: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "method",
                "n",
                "state_dim",
                "N",
                "seed",
                "error",
                "T",
                "rho_q",
                "rho_v",
                "controller",
                "metric_name",
                "metric_implementation",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    "robotarm_dim_scaling",
                    row.method,
                    row.n,
                    row.state_dim,
                    row.budget,
                    row.seed,
                    f"{row.error:.12g}",
                    T,
                    rho_q,
                    rho_v,
                    controller,
                    METRIC_NAME,
                    METRIC_IMPLEMENTATION,
                ]
            )


def load_results_csv(
    path: Path,
    n_values: tuple[int, ...],
    budgets: tuple[int, ...],
    n_seeds: int,
    T: float,
    rho_q: float,
    rho_v: float,
    controller: str,
) -> list[ExperimentResult] | None:
    if not path.exists():
        return None

    results: list[ExperimentResult] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("experiment") != "robotarm_dim_scaling":
                continue
            if not (
                np.isclose(float(row.get("T", np.nan)), T)
                and np.isclose(float(row.get("rho_q", np.nan)), rho_q)
                and np.isclose(float(row.get("rho_v", np.nan)), rho_v)
                and row.get("controller", row.get("controller_mode")) == controller
                and row.get("metric_name") == METRIC_NAME
                and row.get("metric_implementation") == METRIC_IMPLEMENTATION
            ):
                return None
            results.append(
                ExperimentResult(
                    method=row["method"],
                    n=int(row["n"]),
                    state_dim=int(row["state_dim"]),
                    budget=int(row["N"]),
                    seed=int(row["seed"]),
                    error=float(row["error"]),
                )
            )

    by_combo = {}
    for row in results:
        if row.method != "uniform":
            continue
        by_combo.setdefault((row.n, row.budget), 0)
        by_combo[(row.n, row.budget)] += 1
    for n in n_values:
        for budget in budgets:
            if by_combo.get((n, budget), 0) < n_seeds:
                return None
    return results


def aggregate_results(
    results: list[ExperimentResult],
    n: int,
    budgets: tuple[int, ...],
    n_seeds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((len(budgets), n_seeds), dtype=float)
    for i, budget in enumerate(budgets):
        selected = [
            r.error
            for r in results
            if r.method == "uniform" and r.n == n and r.budget == budget
        ]
        values[i] = np.asarray(selected[:n_seeds], dtype=float)
    return mean_and_ci(values)


def plot_results(
    results: list[ExperimentResult],
    n_values: tuple[int, ...],
    budgets: tuple[int, ...],
    n_seeds: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(n_values)))
    markers = ("o", "s", "D", "^")

    for n, color, marker in zip(n_values, colors, markers):
        mean, lo, hi = aggregate_results(results, n, budgets, n_seeds)
        ax.plot(budgets, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}, dim={2 * n}")
        ax.fill_between(budgets, lo, hi, color=color, alpha=0.20)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of samples N", fontsize=LABEL_SIZE)
    ax.set_ylabel("Hausdorff distance", fontsize=LABEL_SIZE)
    ax.set_title("Robot-Arm Uniform Sampling", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    n_values = parse_int_tuple(args.n_values)
    budgets = parse_int_tuple(args.budgets)
    if any(n not in DEFAULT_LINK_COUNTS for n in n_values):
        raise ValueError("Only n=2, n=3, and n=4 are supported for this tracking experiment.")
    if args.n_seeds <= 0 or args.N_ref <= 0 or args.coverage_subset <= 0:
        raise ValueError("n_seeds, N_ref, and coverage_subset must be positive.")
    controller = controller_label(
        args.kp,
        args.kd,
        args.lambda_gain,
        args.gamma,
        args.sigma,
        args.tau_limit,
    )

    results = None if args.force else load_results_csv(
        args.csv,
        n_values,
        budgets,
        args.n_seeds,
        args.T,
        args.rho_q,
        args.rho_v,
        controller,
    )
    if results is None:
        results = run_experiment(
            n_values,
            budgets,
            args.n_seeds,
            args.N_ref,
            args.coverage_subset,
            args.T,
            args.rho_q,
            args.rho_v,
            args.kp,
            args.kd,
            args.lambda_gain,
            args.gamma,
            args.sigma,
            args.tau_limit,
        )
        save_results_csv(results, args.csv, args.T, args.rho_q, args.rho_v, controller)
    else:
        print(f"Loaded cached robot-arm dimension-scaling results from {args.csv}")

    plot_results(results, n_values, budgets, args.n_seeds, args.fig)
    print("\nSaved:")
    print(args.fig)
    print(args.csv)


if __name__ == "__main__":
    main()
