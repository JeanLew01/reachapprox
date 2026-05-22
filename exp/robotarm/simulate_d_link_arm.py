import time
import mujoco
import mujoco.viewer
import numpy as np


def make_d_link_arm_xml(d: int, link_length: float = 0.5) -> str:
    assert d >= 1

    option = """
  <option timestep="0.002" gravity="0 0 0"/>
"""

    asset = """
  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.15 0.25 0.35"
             rgb2="0.55 0.65 0.75"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8"
              reflectance="0.1"/>
  </asset>
"""

    worldbody_start = """
  <worldbody>
    <light name="top_light" pos="0 0 3" diffuse="1 1 1" ambient="0.6 0.6 0.6"/>
    <camera name="fixed" pos="0 0 3.0" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" pos="0 0 -0.02"
          size="3 3 0.01" material="grid_mat"/>
"""

    body_xml = ""
    indent = "    "
    for i in range(d):
        body_name = f"link{i+1}"
        joint_name = f"joint{i+1}"
        geom_name = f"geom{i+1}"
        pos = "0 0 0" if i == 0 else f"{link_length} 0 0"

        body_xml += f'{indent}<body name="{body_name}" pos="{pos}">\n'
        body_xml += (
            f'{indent}  <joint name="{joint_name}" type="hinge" axis="0 0 1" '
            f'damping="0.2" armature="0.01"/>\n'
        )
        body_xml += (
            f'{indent}  <geom name="{geom_name}" type="capsule" '
            f'fromto="0 0 0 {link_length} 0 0" '
            f'size="0.03" density="1000" rgba="0.1 0.3 0.8 1"/>\n'
        )
        indent += "  "

    for _ in range(d):
        indent = indent[:-2]
        body_xml += f"{indent}</body>\n"

    worldbody_end = "  </worldbody>\n"

    actuator_xml = "  <actuator>\n"
    for i in range(d):
        actuator_xml += f'    <motor name="motor{i+1}" joint="joint{i+1}" gear="1"/>\n'
    actuator_xml += "  </actuator>\n"

    xml = f"""
<mujoco model="{d}_link_planar_arm">
{option}
{asset}
{worldbody_start}
{body_xml}
{worldbody_end}
{actuator_xml}
</mujoco>
"""
    return xml


def viewer_demo(d=3, T=5.0):
    xml = make_d_link_arm_xml(d)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    q_goal = np.linspace(0.4, 0.8, d)
    Kp = np.diag([80.0] * d)
    Kd = np.diag([12.0] * d)

    data.qpos[:d] = 0.0
    data.qvel[:d] = 0.0
    mujoco.mj_forward(model, data)

    num_steps = int(T / model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(num_steps):
            q = data.qpos[:d].copy()
            v = data.qvel[:d].copy()
            tau = Kp @ (q_goal - q) - Kd @ v
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        print("q(T) =", data.qpos[:d])
        print("v(T) =", data.qvel[:d])

        time.sleep(5)


if __name__ == "__main__":
    viewer_demo(d=3, T=5.0)