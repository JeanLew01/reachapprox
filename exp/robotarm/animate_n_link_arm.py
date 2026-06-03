"""Interactive MuJoCo viewer for the vertical planar n-link arm."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from fun.mujoco_n_link_arm import (
    DEFAULT_GAMMA,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_LAMBDA,
    DEFAULT_SIGMA,
    DEFAULT_TAU_LIMIT,
    MuJoCoNLinkArm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    parser.add_argument("--lambda_gain", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--tau_limit", type=float, default=DEFAULT_TAU_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm = MuJoCoNLinkArm(
        n=args.n,
        T=args.T,
        kp=args.kp,
        kd=args.kd,
        lambda_gain=args.lambda_gain,
        gamma=args.gamma,
        sigma=args.sigma,
        tau_limit=args.tau_limit,
    )
    arm.reset(np.zeros(args.n), np.zeros(args.n))

    try:
        with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
            end_time = arm.data.time + args.T
            while viewer.is_running() and arm.data.time < end_time:
                start = time.time()
                arm.step()
                viewer.sync()
                elapsed = time.time() - start
                time.sleep(max(0.0, arm.model.opt.timestep - elapsed))

            print(f"q(T) = {arm.data.qpos[:args.n].copy()}")
            print(f"v(T) = {arm.data.qvel[:args.n].copy()}")
            print("Rollout complete. Close the viewer window when finished inspecting.")
            while viewer.is_running():
                mujoco.mj_forward(arm.model, arm.data)
                viewer.sync()
                time.sleep(0.05)
    except Exception as exc:
        print(f"Warning: MuJoCo viewer failed to launch or run: {exc}")


if __name__ == "__main__":
    main()
