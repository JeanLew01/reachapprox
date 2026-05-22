import mujoco
import numpy as np
import imageio.v2 as imageio


def make_d_link_arm_xml(d: int, link_length: float = 0.5) -> str:
    """
    Generate a planar d-link serial arm in MuJoCo.
    Each joint is a hinge joint around the z-axis.
    The arm moves in the xy-plane.
    """
    assert d >= 1

    option = """
  <option timestep="0.002" gravity="0 0 0"/>
  <visual>
    <global azimuth="90" elevation="-90"/>
  </visual>
"""

    worldbody_start = """
  <worldbody>
    <light name="top_light" pos="0 0 3"/>
    <camera name="fixed" pos="0 0 3.0" xyaxes="1 0 0 0 1 0"/>
"""

    # Build nested body chain.
    body_xml = ""
    indent = "    "
    for i in range(d):
        body_name = f"link{i+1}"
        joint_name = f"joint{i+1}"
        geom_name = f"geom{i+1}"

        # First link starts at world origin.
        # Later links are attached at the end of the previous link.
        pos = "0 0 0" if i == 0 else f"{link_length} 0 0"

        body_xml += f'{indent}<body name="{body_name}" pos="{pos}">\n'
        body_xml += f'{indent}  <joint name="{joint_name}" type="hinge" axis="0 0 1"/>\n'
        body_xml += (
            f'{indent}  <geom name="{geom_name}" type="capsule" '
            f'fromto="0 0 0 {link_length} 0 0" '
            f'size="0.03" density="1000"/>\n'
        )

        indent += "  "

    # Close nested bodies.
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
{worldbody_start}
{body_xml}
{worldbody_end}
{actuator_xml}
</mujoco>
"""
    return xml


def rollout_arm(d: int, T: float = 2.0, render: bool = True):
    xml = make_d_link_arm_xml(d)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Initial condition.
    q0 = np.zeros(d)
    v0 = np.zeros(d)

    # Goal configuration. You can change this later.
    q_goal = np.linspace(0.4, 0.8, d)

    # Diagonal PD gains.
    Kp = np.diag([25.0] * d)
    Kd = np.diag([4.0] * d)

    data.qpos[:d] = q0
    data.qvel[:d] = v0
    mujoco.mj_forward(model, data)

    num_steps = int(T / model.opt.timestep)

    q_traj = []
    v_traj = []

    for _ in range(num_steps):
        q = data.qpos[:d].copy()
        v = data.qvel[:d].copy()

        tau = Kp @ (q_goal - q) - Kd @ v
        data.ctrl[:] = tau

        mujoco.mj_step(model, data)

        q_traj.append(data.qpos[:d].copy())
        v_traj.append(data.qvel[:d].copy())

    qT = data.qpos[:d].copy()
    vT = data.qvel[:d].copy()

    print(f"d = {d}")
    print("state dimension n = ", 2 * d)
    print("q_goal =", q_goal)
    print("q(T)   =", qT)
    print("v(T)   =", vT)

    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera="fixed")
        img = renderer.render()
        filename = f"{d}_link_arm_final.png"
        imageio.imwrite(filename, img)
        print(f"Saved final image to {filename}")

    return np.array(q_traj), np.array(v_traj)


if __name__ == "__main__":
    # Change d to 2 or 3.
    rollout_arm(d=2, T=2.0, render=True)
    rollout_arm(d=3, T=2.0, render=True)