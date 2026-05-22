"""Generate closed-loop endpoint samples for the MuJoCo n-link arm benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mujoco_n_link_arm import MuJoCoNLinkArm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(2, 3), required=True)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--rho_q", type=float, default=0.1)
    parser.add_argument("--rho_v", type=float, default=0.1)
    parser.add_argument(
        "--controller_mode",
        choices=("pd", "gravity_compensated_pd"),
        default="pd",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N <= 0:
        raise ValueError("--N must be positive.")
    if args.rho_q < 0.0 or args.rho_v < 0.0:
        raise ValueError("--rho_q and --rho_v must be nonnegative.")

    rng = np.random.default_rng(args.seed)
    n = args.n
    q_goal = np.linspace(0.4, 0.8, n)
    q0_nom = np.zeros(n)
    v0_nom = np.zeros(n)

    q0 = q0_nom + rng.uniform(-args.rho_q, args.rho_q, size=(args.N, n))
    v0 = v0_nom + rng.uniform(-args.rho_v, args.rho_v, size=(args.N, n))
    X0 = np.hstack([q0, v0])
    XT = np.empty_like(X0)

    arm = MuJoCoNLinkArm(n=n, T=args.T, controller_mode=args.controller_mode)
    for i, x0 in enumerate(X0):
        XT[i] = arm.rollout(x0, q_goal, T=args.T)
        if (i + 1) % max(1, args.N // 10) == 0:
            print(f"completed {i + 1}/{args.N} rollouts")

    out = Path(args.out) if args.out else Path(f"robotarm_n{n}_endpoints.npz")
    np.savez(
        out,
        X0=X0,
        XT=XT,
        q_goal=q_goal,
        n=n,
        T=args.T,
        rho_q=args.rho_q,
        rho_v=args.rho_v,
        controller_mode=args.controller_mode,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
