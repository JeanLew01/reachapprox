"""Adversarial sampling for robot-arm uncertainty propagation.

The adversarial sampler follows the update-count convention used in the other
experiments.  With ``n_adv=1``, it samples a smaller batch of initial particles,
propagates them, performs one endpoint-repulsion update projected back to the
initial box, and returns the accumulated endpoints.
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
    MuJoCoNLinkArm,
)
from reachapprox.exp.robotarm.fun.dim_scaling import (  # noqa: E402
    EXPERIMENT_SEED,
    METRIC_IMPLEMENTATION,
    METRIC_NAME,
    REFERENCE_SEED,
    RHO_Q,
    T_HORIZON,
    directed_hausdorff_to_convex_hull,
    parse_int_tuple,
    sample_initial_box,
)


DEFAULT_LINK_COUNTS = (2, 3, 4)
DEFAULT_SAMPLE_BUDGETS = (1, 3, 10, 30, 100, 300, 1000, 3000)
DEFAULT_N_SEEDS = 10
DEFAULT_N_REF = 20_000
DEFAULT_COVERAGE_SUBSET = 2_000
DEFAULT_N_ADV = 1
DEFAULT_ETA = 0.20
DEFAULT_LAMBDA_REG = 1e-4
RHO_V = 0.1

OUT_DIR = Path("reachapprox/exp/robotarm/results")
COMBINED_CSV = OUT_DIR / "robotarm_adversarial_nadv1_dim_scaling_n234.csv"
COMBINED_FIG = OUT_DIR / "robotarm_adversarial_nadv1_dim_scaling_n234.png"
CORL_CSV = Path("CoRL_2026/fig/robotarm_adversarial_nadv1_dim_scaling_results_n234.csv")
CORL_FIG = Path("CoRL_2026/fig/robotarm_adversarial_nadv1_dim_scaling_hausdorff_vs_N_n234.png")


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
    parser.add_argument("--n_values", default="2,3,4")
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
    parser.add_argument("--n_adv", type=int, default=DEFAULT_N_ADV)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--lambda_reg", type=float, default=DEFAULT_LAMBDA_REG)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--combined_csv", type=Path, default=COMBINED_CSV)
    parser.add_argument("--combined_fig", type=Path, default=COMBINED_FIG)
    parser.add_argument("--corl_csv", type=Path, default=CORL_CSV)
    parser.add_argument("--corl_fig", type=Path, default=CORL_FIG)
    return parser.parse_args()


def controller_label(args: argparse.Namespace) -> str:
    return (
        f"{TRACKING_CONTROLLER_NAME}(kp={args.kp:.6g},kd={args.kd:.6g},"
        f"lambda={args.lambda_gain:.6g},gamma={args.gamma:.6g},"
        f"sigma={args.sigma:.6g},tau_limit={args.tau_limit:.6g})"
    )


def propagate_endpoints(
    X0: np.ndarray,
    n: int,
    args: argparse.Namespace,
) -> np.ndarray:
    arm = MuJoCoNLinkArm(
        n=n,
        T=args.T,
        kp=args.kp,
        kd=args.kd,
        lambda_gain=args.lambda_gain,
        gamma=args.gamma,
        sigma=args.sigma,
        tau_limit=args.tau_limit,
    )
    XT = np.empty_like(X0)
    for i, x0 in enumerate(X0):
        XT[i] = arm.rollout(x0, T=args.T)
    return XT


def regularized_inverse_covariance(Y: np.ndarray, lambda_reg: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(Y, axis=0)
    centered = Y - center
    if Y.shape[0] > 1:
        cov = (centered.T @ centered) / (Y.shape[0] - 1)
    else:
        cov = np.zeros((Y.shape[1], Y.shape[1]), dtype=float)
    Q = np.linalg.inv(cov + lambda_reg * np.eye(Y.shape[1]))
    return center, Q


def project_initial_box(X: np.ndarray, n: int, rho_q: float, rho_v: float) -> np.ndarray:
    projected = np.asarray(X, dtype=float).copy()
    projected[:, :n] = np.clip(projected[:, :n], -rho_q, rho_q)
    projected[:, n:] = np.clip(projected[:, n:], -rho_v, rho_v)
    return projected


def adversarial_endpoint_samples(
    rng: np.random.Generator,
    total_budget: int,
    n: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Generate exactly N adversarial endpoints with n_adv projected updates."""

    if total_budget == 1:
        X = sample_initial_box(rng, 1, n, args.rho_q, args.rho_v)
        return propagate_endpoints(X, n, args)

    particle_count = int(np.ceil(total_budget / (args.n_adv + 1)))
    X = sample_initial_box(rng, particle_count, n, args.rho_q, args.rho_v)
    endpoint_batches = [propagate_endpoints(X, n, args)]

    for _ in range(args.n_adv):
        Y_acc = np.vstack(endpoint_batches)
        center, Q = regularized_inverse_covariance(Y_acc, args.lambda_reg)
        Y_current = propagate_endpoints(X, n, args)
        grad = 2.0 * (Y_current - center) @ Q.T
        grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
        X = project_initial_box(X + args.eta * grad, n, args.rho_q, args.rho_v)
        endpoint_batches.append(propagate_endpoints(X, n, args))

    return np.vstack(endpoint_batches)[:total_budget]


def mean_and_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=1)
    if values.shape[1] == 1:
        return mean, mean, mean
    stderr = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
    half_width = 1.96 * stderr
    return mean, np.maximum(mean - half_width, np.finfo(float).tiny), mean + half_width


def run_experiment(
    n_values: tuple[int, ...],
    budgets: tuple[int, ...],
    args: argparse.Namespace,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    max_budget = max(budgets)

    for n in n_values:
        ref_rng = np.random.default_rng(REFERENCE_SEED + n)
        X_ref = sample_initial_box(ref_rng, args.N_ref, n, args.rho_q, args.rho_v)
        Y_ref = propagate_endpoints(X_ref, n, args)
        subset_size = min(args.coverage_subset, args.N_ref)
        print(f"\nRobot arm n={n}, state_dim={2 * n}: reference {args.N_ref}, subset {subset_size}")

        for seed_index in range(args.n_seeds):
            seed = EXPERIMENT_SEED + 500_000 + 10_000 * n + 100 * seed_index
            rng = np.random.default_rng(seed)
            subset_idx = rng.choice(args.N_ref, size=subset_size, replace=False)
            Y_ref_subset = Y_ref[subset_idx]

            for budget in budgets:
                Y_adv = adversarial_endpoint_samples(rng, budget, n, args)
                metric_rng = np.random.default_rng(seed + budget + 9_000_000)
                error = directed_hausdorff_to_convex_hull(Y_ref_subset, Y_adv, metric_rng)
                results.append(
                    ExperimentResult(f"adversarial_nadv{args.n_adv}", n, 2 * n, budget, seed + budget, error)
                )
            print(f"  finished seed {seed_index + 1}/{args.n_seeds}")

        n_results = [row for row in results if row.n == n]
        save_results_csv(
            n_results,
            args.out_dir / f"robotarm_adversarial_nadv{args.n_adv}_dim_scaling_n{n}.csv",
            args,
        )
        plot_results(
            n_results,
            (n,),
            budgets,
            args.n_seeds,
            args.out_dir / f"robotarm_adversarial_nadv{args.n_adv}_dim_scaling_n{n}.png",
        )
        print(f"  saved n={n} adversarial results")

    return results


def save_results_csv(results: list[ExperimentResult], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
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
                    args.T,
                    args.rho_q,
                    args.rho_v,
                    controller_label(args),
                    METRIC_NAME,
                    METRIC_IMPLEMENTATION,
                ]
            )


def aggregate(
    results: list[ExperimentResult],
    n: int,
    budgets: tuple[int, ...],
    n_seeds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((len(budgets), n_seeds), dtype=float)
    for i, budget in enumerate(budgets):
        selected = [r.error for r in results if r.n == n and r.budget == budget]
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
        mean, lo, hi = aggregate(results, n, budgets, n_seeds)
        ax.plot(budgets, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}, dim={2 * n}")
        ax.fill_between(budgets, lo, hi, color=color, alpha=0.20)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of samples N", fontsize=18)
    ax.set_ylabel("Hausdorff Distance", fontsize=18)
    ax.set_title("Robot-Arm Adversarial Sampling", fontsize=20)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=12)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_slopes(results: list[ExperimentResult], n_values: tuple[int, ...], budgets: tuple[int, ...]) -> None:
    print("\nAdversarial robot-arm slopes")
    for n in n_values:
        means = np.array(
            [
                np.mean([r.error for r in results if r.n == n and r.budget == budget])
                for budget in budgets
            ],
            dtype=float,
        )
        slope_all = float(np.polyfit(np.log(budgets), np.log(means), 1)[0])
        mask = np.asarray(budgets) >= 10
        slope_tail = (
            float(np.polyfit(np.log(np.asarray(budgets)[mask]), np.log(means[mask]), 1)[0])
            if np.count_nonzero(mask) >= 2
            else float("nan")
        )
        print(
            f"n={n}, dim={2 * n}, slope_all={slope_all:.4f}, "
            f"slope_N>=10={slope_tail:.4f}, target={-1 / (2 * n):.4f}"
        )


def main() -> None:
    args = parse_args()
    n_values = parse_int_tuple(args.n_values)
    budgets = parse_int_tuple(args.budgets)
    if any(n not in DEFAULT_LINK_COUNTS for n in n_values):
        raise ValueError("Only n=2, n=3, and n=4 are supported.")
    if max(budgets) <= 0:
        raise ValueError("Budgets must be positive.")
    if args.n_adv < 0:
        raise ValueError("n_adv must be nonnegative.")
    if args.eta <= 0.0:
        raise ValueError("eta must be positive.")
    if args.lambda_reg <= 0.0:
        raise ValueError("lambda_reg must be positive.")

    results = run_experiment(n_values, budgets, args)
    save_results_csv(results, args.combined_csv, args)
    save_results_csv(results, args.corl_csv, args)
    plot_results(results, n_values, budgets, args.n_seeds, args.combined_fig)
    plot_results(results, n_values, budgets, args.n_seeds, args.corl_fig)
    print_slopes(results, n_values, budgets)
    print("\nSaved:")
    print(args.combined_fig)
    print(args.combined_csv)
    print(args.corl_fig)
    print(args.corl_csv)


if __name__ == "__main__":
    main()
