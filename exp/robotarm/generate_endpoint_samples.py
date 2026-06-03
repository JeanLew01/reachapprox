"""Generate closed-loop endpoint samples for the MuJoCo n-link arm benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fun.mujoco_n_link_arm import (
    DEFAULT_GAMMA,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_LAMBDA,
    DEFAULT_SIGMA,
    DEFAULT_TAU_LIMIT,
    TRACKING_CONTROLLER_NAME,
    MuJoCoNLinkArm,
    reference_parameters,
    reference_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--rho_q", type=float, default=0.1)
    parser.add_argument("--rho_v", type=float, default=0.1)
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    parser.add_argument("--lambda_gain", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--tau_limit", type=float, default=DEFAULT_TAU_LIMIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def make_arm(args: argparse.Namespace) -> MuJoCoNLinkArm:
    return MuJoCoNLinkArm(
        n=args.n,
        T=args.T,
        kp=args.kp,
        kd=args.kd,
        lambda_gain=args.lambda_gain,
        gamma=args.gamma,
        sigma=args.sigma,
        tau_limit=args.tau_limit,
    )


def save_nominal_diagnostic(args: argparse.Namespace, out_dir: Path) -> None:
    arm = make_arm(args)
    arm.reset(np.zeros(args.n), np.zeros(args.n))
    times: list[float] = []
    q_hist: list[np.ndarray] = []
    qd_hist: list[np.ndarray] = []

    while arm.data.time < args.T - 0.5 * arm.model.opt.timestep:
        t = float(arm.data.time)
        qd, _, _ = reference_trajectory(t, args.n)
        times.append(t)
        q_hist.append(arm.data.qpos[: args.n].copy())
        qd_hist.append(qd)
        arm.step()

    t = float(arm.data.time)
    qd, _, _ = reference_trajectory(t, args.n)
    times.append(t)
    q_hist.append(arm.data.qpos[: args.n].copy())
    qd_hist.append(qd)

    times_arr = np.asarray(times)
    q_arr = np.vstack(q_hist)
    qd_arr = np.vstack(qd_hist)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), constrained_layout=True, sharex=True)
    for i in range(args.n):
        axes[0].plot(times_arr, q_arr[:, i], lw=2.0, label=f"$q_{i + 1}$")
        axes[0].plot(times_arr, qd_arr[:, i], lw=1.5, linestyle="--", label=f"$q_{{d,{i + 1}}}$")
        axes[1].plot(times_arr, q_arr[:, i] - qd_arr[:, i], lw=2.0, label=f"$q_{i + 1}-q_{{d,{i + 1}}}$")
    axes[0].set_ylabel("joint angle")
    axes[0].set_title(f"Adaptive ID tracking diagnostic, n={args.n}")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("tracking error")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(frameon=False, ncol=2)
    fig.savefig(out_dir / f"arm_tracking_n{args.n}_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_terminal_sample_plot(XT: np.ndarray, n: int, out_dir: Path) -> None:
    if n == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
        axes[0].scatter(XT[:, 0], XT[:, 1], s=6, alpha=0.42, linewidths=0)
        axes[0].set_xlabel("$q_1(T)$")
        axes[0].set_ylabel("$q_2(T)$")
        axes[1].scatter(XT[:, 2], XT[:, 3], s=6, alpha=0.42, linewidths=0)
        axes[1].set_xlabel("$\\dot q_1(T)$")
        axes[1].set_ylabel("$\\dot q_2(T)$")
    elif n in (3, 4):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
        axes[0].scatter(XT[:, 0], XT[:, 1], s=6, alpha=0.42, linewidths=0)
        axes[0].set_xlabel("$q_1(T)$")
        axes[0].set_ylabel("$q_2(T)$")
        axes[1].scatter(XT[:, 1], XT[:, 2], s=6, alpha=0.42, linewidths=0)
        axes[1].set_xlabel("$q_2(T)$")
        axes[1].set_ylabel("$q_3(T)$")
        axes[2].scatter(XT[:, n], XT[:, n + 1], s=6, alpha=0.42, linewidths=0)
        axes[2].set_xlabel("$\\dot q_1(T)$")
        axes[2].set_ylabel("$\\dot q_2(T)$")
    else:
        raise ValueError(f"Unsupported n={n}.")

    for ax in axes:
        ax.grid(True, alpha=0.28)
    fig.suptitle(f"Terminal samples, n={n}")
    fig.savefig(out_dir / f"arm_tracking_n{n}_terminal_samples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.N <= 0:
        raise ValueError("--N must be positive.")
    if args.rho_q < 0.0 or args.rho_v < 0.0:
        raise ValueError("--rho_q and --rho_v must be nonnegative.")

    rng = np.random.default_rng(args.seed)
    n = args.n
    q0_nom = np.zeros(n)
    v0_nom = np.zeros(n)

    q0 = q0_nom + rng.uniform(-args.rho_q, args.rho_q, size=(args.N, n))
    v0 = v0_nom + rng.uniform(-args.rho_v, args.rho_v, size=(args.N, n))
    X0 = np.hstack([q0, v0])
    XT = np.empty_like(X0)

    arm = make_arm(args)
    for i, x0 in enumerate(X0):
        XT[i] = arm.rollout(x0, T=args.T)
        if (i + 1) % max(1, args.N // 10) == 0:
            print(f"completed {i + 1}/{args.N} rollouts")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"arm_tracking_n{n}_terminal_samples.npz"
    params = reference_parameters(n)
    ref_array = np.vstack(
        [
            np.asarray(params["q_c"], dtype=float),
            np.asarray(params["A"], dtype=float),
            float(params["omega"]) * np.ones(n),
            np.asarray(params["phi"], dtype=float),
        ]
    )
    np.savez(
        out,
        X0=X0,
        XT=XT,
        n=n,
        T=args.T,
        rho_q=args.rho_q,
        rho_v=args.rho_v,
        controller=arm.controller.description(),
        controller_name=TRACKING_CONTROLLER_NAME,
        reference_parameters=ref_array,
        q_c=params["q_c"],
        A=params["A"],
        omega=params["omega"],
        phi=params["phi"],
    )
    print(f"saved {out}")
    if not args.skip_plots:
        save_nominal_diagnostic(args, out_dir)
        save_terminal_sample_plot(XT, n, out_dir)
        print(f"saved {out_dir / f'arm_tracking_n{n}_diagnostic.png'}")
        print(f"saved {out_dir / f'arm_tracking_n{n}_terminal_samples.png'}")


if __name__ == "__main__":
    main()
