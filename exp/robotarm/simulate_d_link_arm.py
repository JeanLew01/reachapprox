import time
import mujoco
import mujoco.viewer
import numpy as np

from fun.mujoco_n_link_arm import MuJoCoNLinkArm, make_n_link_arm_xml


def make_d_link_arm_xml(d: int, link_length: float = 0.5) -> str:
    return make_n_link_arm_xml(d, link_length=link_length)


def viewer_demo(d=3, T=5.0):
    arm = MuJoCoNLinkArm(n=d, T=T)
    arm.reset(np.zeros(d), np.zeros(d))

    with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
        end_time = arm.data.time + T
        while viewer.is_running() and arm.data.time < end_time:
            arm.step()
            viewer.sync()
            time.sleep(arm.model.opt.timestep)

        print("q(T) =", arm.data.qpos[:d])
        print("v(T) =", arm.data.qvel[:d])

        time.sleep(5)


if __name__ == "__main__":
    viewer_demo(d=3, T=5.0)
