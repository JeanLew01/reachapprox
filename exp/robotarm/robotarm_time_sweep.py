"""Time sweep for robot-arm point-cloud Hausdorff error.

This experiment fixes the sample budget and measures how the terminal
point-cloud directed Hausdorff error changes with propagation time.
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

from reachapprox.exp.robotarm.fun.mujoco_n_link_arm import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_LAMBDA,
    DEFAULT_SIGMA,
    DEFAULT_TAU_LIMIT,
    TRACKING_CONTROLLER_NAME,
)
from reachapprox.exp.robotarm.fun.dim_scaling import (  # noqa: E402
    EXPERIMENT_SEED,
    METRIC_IMPLEMENTATION,
    METRIC_NAME,
    REFERENCE_SEED,
    RHO_Q,
    RHO_V,
    controller_label,
    directed_hausdorff_to_convex_hull,
    parse_int_tuple,
    propagate_endpoints,
    sample_initial_box,
)


DEFAULT_LINK_COUNTS = (2, 3, 4)
DEFAULT_TIME_GRID = tuple(float(t) for t in np.linspace(0.01, 1.0, 10))
DEFAULT_BUDGET = 1000
DEFAULT_N_SEEDS = 5
DEFAULT_N_REF = 5000
DEFAULT_COVERAGE_SUBSET = 1000

OUT_DIR = Path("reachapprox/exp/robotarm/results")
CSV_PATH = OUT_DIR / "robotarm_time_sweep_results.csv"
FIG_PATH = OUT_DIR / "robotarm_time_sweep_hausdorff_vs_time.png"
CORL_CSV_PATH = Path("CoRL_2026/fig/robotarm_time_sweep_results.csv")
CORL_FIG_PATH = Path("CoRL_2026/fig/robotarm_time_sweep_hausdorff_vs_time.png")


@dataclass(frozen=True)
class TimeSweepResult:
    method: str
    n: int
    state_dim: int
    budget: int
    time: float
    seed: int
    error: float


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_values", default="2,3,4")
    parser.add_argument("--times", default=",".join(f"{t:.6g}" for t in DEFAULT_TIME_GRID))
    parser.add_argument("--N", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--n_seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--N_ref", type=int, default=DEFAULT_N_REF)
    parser.add_argument("--coverage_subset", type=int, default=DEFAULT_COVERAGE_SUBSET)
    parser.add_argument("--rho_q", type=float, default=RHO_Q)
    parser.add_argument("--rho_v", type=float, default=RHO_V)
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    parser.add_argument("--lambda_gain", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--tau_limit", type=float, default=DEFAULT_TAU_LIMIT)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--fig", type=Path, default=FIG_PATH)
    parser.add_argument("--corl_csv", type=Path, default=CORL_CSV_PATH)
    parser.add_argument("--corl_fig", type=Path, default=CORL_FIG_PATH)
    return parser.parse_args()


def run_experiment(
    n_values: tuple[int, ...],
    times: tuple[float, ...],
    args: argparse.Namespace,
) -> list[TimeSweepResult]:
    results: list[TimeSweepResult] = []

    for n in n_values:
        ref_rng = np.random.default_rng(REFERENCE_SEED + 30_000 + n)
        X_ref = sample_initial_box(ref_rng, args.N_ref, n, args.rho_q, args.rho_v)
        print(f"\nRobot arm n={n}, state_dim={2 * n}, N={args.N}")

        for time in times:
            Y_ref = propagate_endpoints(
                X_ref,
                n,
                time,
                args.kp,
                args.kd,
                args.lambda_gain,
                args.gamma,
                args.sigma,
                args.tau_limit,
            )
            subset_size = min(args.coverage_subset, args.N_ref)

            for seed_index in range(args.n_seeds):
                seed = EXPERIMENT_SEED + 700_000 + 10_000 * n + 100 * seed_index + int(round(1000 * time))
                rng = np.random.default_rng(seed)
                subset_idx = rng.choice(args.N_ref, size=subset_size, replace=False)
                Y_ref_subset = Y_ref[subset_idx]
                X_sample = sample_initial_box(rng, args.N, n, args.rho_q, args.rho_v)
                Y_sample = propagate_endpoints(
                    X_sample,
                    n,
                    time,
                    args.kp,
                    args.kd,
                    args.lambda_gain,
                    args.gamma,
                    args.sigma,
                    args.tau_limit,
                )
                metric_rng = np.random.default_rng(seed + 9_000_000)
                error = directed_hausdorff_to_convex_hull(Y_ref_subset, Y_sample, metric_rng)
                results.append(TimeSweepResult("uniform", n, 2 * n, args.N, time, seed, error))

            print(f"  finished T={time:.4f}")

    return results


def save_results_csv(results: list[TimeSweepResult], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = controller_label(args.kp, args.kd, args.lambda_gain, args.gamma, args.sigma, args.tau_limit)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "method",
                "n",
                "state_dim",
                "N",
                "time",
                "seed",
                "error",
                "rho_q",
                "rho_v",
                "controller",
                "controller_name",
                "metric_name",
                "metric_implementation",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    "robotarm_time_sweep",
                    row.method,
                    row.n,
                    row.state_dim,
                    row.budget,
                    f"{row.time:.12g}",
                    row.seed,
                    f"{row.error:.12g}",
                    args.rho_q,
                    args.rho_v,
                    label,
                    TRACKING_CONTROLLER_NAME,
                    METRIC_NAME,
                    METRIC_IMPLEMENTATION,
                ]
            )


def aggregate(
    results: list[TimeSweepResult],
    n: int,
    times: tuple[float, ...],
    n_seeds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((len(times), n_seeds), dtype=float)
    for i, time in enumerate(times):
        selected = [
            r.error
            for r in results
            if r.n == n and np.isclose(r.time, time)
        ]
        values[i] = np.asarray(selected[:n_seeds], dtype=float)
    mean = values.mean(axis=1)
    if n_seeds == 1:
        return mean, mean, mean
    stderr = values.std(axis=1, ddof=1) / np.sqrt(n_seeds)
    half_width = 1.96 * stderr
    return mean, np.maximum(mean - half_width, np.finfo(float).tiny), mean + half_width


def plot_results(
    results: list[TimeSweepResult],
    n_values: tuple[int, ...],
    times: tuple[float, ...],
    n_seeds: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(n_values)))
    markers = ("o", "s", "D", "^")
    times_arr = np.asarray(times, dtype=float)

    for n, color, marker in zip(n_values, colors, markers):
        mean, lo, hi = aggregate(results, n, times, n_seeds)
        ax.plot(times_arr, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}, dim={2 * n}")
        ax.fill_between(times_arr, lo, hi, color=color, alpha=0.20)

    ax.set_xlabel("time T", fontsize=18)
    ax.set_ylabel("Hausdorff distance", fontsize=18)
    ax.set_title("Robot-Arm Time Sweep", fontsize=20)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=12)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(results: list[TimeSweepResult], n_values: tuple[int, ...], times: tuple[float, ...]) -> None:
    print("\nRobot-arm time-sweep means")
    for n in n_values:
        means = []
        for time in times:
            vals = [r.error for r in results if r.n == n and np.isclose(r.time, time)]
            means.append(float(np.mean(vals)))
        print(f"n={n}: " + ", ".join(f"T={t:.2f}:{m:.6g}" for t, m in zip(times, means)))


def main() -> None:
    args = parse_args()
    n_values = parse_int_tuple(args.n_values)
    times = parse_float_tuple(args.times)
    if any(n not in DEFAULT_LINK_COUNTS for n in n_values):
        raise ValueError("Only n=2, n=3, and n=4 are supported.")
    if args.N <= 0 or args.N_ref <= 0 or args.coverage_subset <= 0 or args.n_seeds <= 0:
        raise ValueError("N, N_ref, coverage_subset, and n_seeds must be positive.")
    if any(time <= 0.0 for time in times):
        raise ValueError("All times must be positive.")

    results = run_experiment(n_values, times, args)
    save_results_csv(results, args.csv, args)
    save_results_csv(results, args.corl_csv, args)
    plot_results(results, n_values, times, args.n_seeds, args.fig)
    plot_results(results, n_values, times, args.n_seeds, args.corl_fig)
    print_summary(results, n_values, times)
    print("\nSaved:")
    print(args.fig)
    print(args.csv)
    print(args.corl_fig)
    print(args.corl_csv)


if __name__ == "__main__":
    main()
