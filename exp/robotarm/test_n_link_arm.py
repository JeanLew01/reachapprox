"""Smoke test for the MuJoCo vertical planar n-link arm."""

from __future__ import annotations

import numpy as np

from fun.mujoco_n_link_arm import MuJoCoNLinkArm


def run_case(n: int, T: float = 1.0) -> None:
    arm = MuJoCoNLinkArm(n=n, T=T)
    q0 = np.zeros(n)
    v0 = np.zeros(n)
    x0 = np.concatenate([q0, v0])
    xT = arm.rollout(x0, T=T)

    print(f"n = {n}")
    print(f"state dimension = {2 * n}")
    print(f"controller = {arm.controller.description()}")
    print(f"q(T) = {xT[:n]}")
    print(f"v(T) = {xT[n:]}")
    print()

    filename = f"{n}_link_arm_final_tracking.png"
    arm.render(filename)


def main() -> None:
    for n in (2, 3, 4):
        run_case(n=n)


if __name__ == "__main__":
    main()
