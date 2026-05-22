"""Interactive MuJoCo viewer for the vertical planar n-link arm."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_n_link_arm import MuJoCoNLinkArm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(2, 3), default=2)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument(
        "--controller_mode",
        choices=("pd", "gravity_compensated_pd"),
        default="pd",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm = MuJoCoNLinkArm(n=args.n, T=args.T, controller_mode=args.controller_mode)
    q_goal = np.linspace(0.4, 0.8, args.n)
    arm.reset(np.zeros(args.n), np.zeros(args.n))

    try:
        with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
            end_time = arm.data.time + args.T
            while viewer.is_running() and arm.data.time < end_time:
                start = time.time()
                arm.step(q_goal)
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
