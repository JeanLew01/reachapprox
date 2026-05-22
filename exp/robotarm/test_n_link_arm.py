"""Smoke test for the MuJoCo vertical planar n-link arm."""

from __future__ import annotations

import numpy as np

from mujoco_n_link_arm import MuJoCoNLinkArm


def run_case(n: int, controller_mode: str, T: float = 2.0) -> None:
    arm = MuJoCoNLinkArm(n=n, T=T, controller_mode=controller_mode)
    q0 = np.zeros(n)
    v0 = np.zeros(n)
    q_goal = np.linspace(0.4, 0.8, n)
    x0 = np.concatenate([q0, v0])
    xT = arm.rollout(x0, q_goal, T=T)

    print(f"n = {n}")
    print(f"state dimension = {2 * n}")
    print(f"controller_mode = {controller_mode}")
    print(f"q_goal = {q_goal}")
    print(f"q(T) = {xT[:n]}")
    print(f"v(T) = {xT[n:]}")
    print()

    filename = f"{n}_link_arm_final_{controller_mode}.png"
    arm.render(filename)


def main() -> None:
    for controller_mode in ("pd", "gravity_compensated_pd"):
        for n in (2, 3):
            run_case(n=n, controller_mode=controller_mode)


if __name__ == "__main__":
    main()
