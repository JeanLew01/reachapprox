"""MuJoCo vertical planar n-link arm dynamics for uncertainty propagation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import mujoco
import numpy as np


SUPPORTED_LINK_COUNTS = (2, 3)


def _validate_n(n: int) -> None:
    if n not in SUPPORTED_LINK_COUNTS:
        raise ValueError(f"Only n={SUPPORTED_LINK_COUNTS} are supported for now; got n={n}.")


def make_n_link_arm_xml(n: int, link_length: float = 0.5, timestep: float = 0.002) -> str:
    """Build a deterministic MJCF model for a vertical planar serial n-link arm.

    The chain lies in the xz-plane and each hinge rotates about the y-axis.
    Gravity is enabled so joint torques include nontrivial gravitational loading.
    """

    _validate_n(n)
    if link_length <= 0.0:
        raise ValueError(f"link_length must be positive; got {link_length}.")
    if timestep <= 0.0:
        raise ValueError(f"timestep must be positive; got {timestep}.")

    option_xml = f'  <option timestep="{timestep:.8g}" gravity="0 0 -9.81"/>\n'
    visual_xml = (
        "  <visual>\n"
        '    <global offwidth="1000" offheight="1000"/>\n'
        "  </visual>\n"
    )
    asset_xml = """
  <asset>
    <texture name="checker" type="2d" builtin="checker"
             rgb1="0.18 0.22 0.25" rgb2="0.65 0.70 0.74"
             width="512" height="512"/>
    <material name="checker_mat" texture="checker" texrepeat="8 8" reflectance="0.1"/>
    <material name="link_mat" rgba="0.10 0.35 0.85 1"/>
  </asset>
"""

    worldbody_xml = """
  <worldbody>
    <light name="key_light" pos="0 -3 3" dir="0 1 -1" diffuse="1 1 1" ambient="0.45 0.45 0.45"/>
    <camera name="fixed" pos="0 -3.0 1.2" xyaxes="1 0 0 0 0 1"/>
    <geom name="ground" type="plane" pos="0 0 -0.55" size="3 3 0.01" material="checker_mat"/>
"""
    indent = "    "
    body_xml = ""
    for i in range(n):
        body_xml += f'{indent}<body name="link{i + 1}" pos="{0.0 if i == 0 else link_length:.8g} 0 0">\n'
        body_xml += (
            f'{indent}  <joint name="joint{i + 1}" type="hinge" axis="0 1 0" '
            'damping="0.2" armature="0.01" limited="false"/>\n'
        )
        body_xml += (
            f'{indent}  <geom name="link{i + 1}_geom" type="capsule" '
            f'fromto="0 0 0 {link_length:.8g} 0 0" size="0.035" '
            'density="1000" material="link_mat" contype="0" conaffinity="0"/>\n'
        )
        indent += "  "

    for _ in range(n):
        indent = indent[:-2]
        body_xml += f"{indent}</body>\n"

    worldbody_xml += body_xml + "  </worldbody>\n"

    actuator_xml = "  <actuator>\n"
    for i in range(n):
        actuator_xml += f'    <motor name="motor{i + 1}" joint="joint{i + 1}" gear="1"/>\n'
    actuator_xml += "  </actuator>\n"

    return (
        f'<mujoco model="vertical_planar_{n}_link_arm">\n'
        f"{option_xml}"
        f"{visual_xml}"
        f"{asset_xml}"
        f"{worldbody_xml}"
        f"{actuator_xml}"
        "</mujoco>\n"
    )


class MuJoCoNLinkArm:
    """Closed-loop MuJoCo simulator for a vertical planar serial n-link arm."""

    def __init__(
        self,
        n: int,
        link_length: float = 0.5,
        timestep: float = 0.002,
        kp: float = 80.0,
        kd: float = 12.0,
        T: float = 2.0,
        controller_mode: str = "pd",
    ) -> None:
        _validate_n(n)
        if controller_mode not in {"pd", "gravity_compensated_pd"}:
            raise ValueError(
                "controller_mode must be 'pd' or 'gravity_compensated_pd'; "
                f"got {controller_mode!r}."
            )
        if T <= 0.0:
            raise ValueError(f"T must be positive; got {T}.")

        self.n = n
        self.link_length = link_length
        self.timestep = timestep
        self.T = T
        self.controller_mode = controller_mode
        self.Kp = kp * np.eye(n)
        self.Kd = kd * np.eye(n)

        xml = make_n_link_arm_xml(n=n, link_length=link_length, timestep=timestep)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

    def _validate_vector(self, name: str, value: np.ndarray, size: int) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},); got {array.shape}.")
        return array

    def reset(self, q0: np.ndarray, v0: np.ndarray) -> None:
        q0 = self._validate_vector("q0", q0, self.n)
        v0 = self._validate_vector("v0", v0, self.n)
        self.data.qpos[: self.n] = q0
        self.data.qvel[: self.n] = v0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> np.ndarray:
        q = self.data.qpos[: self.n].copy()
        v = self.data.qvel[: self.n].copy()
        return np.concatenate([q, v])

    def compute_control(self, q_goal: np.ndarray) -> np.ndarray:
        q_goal = self._validate_vector("q_goal", q_goal, self.n)
        q = self.data.qpos[: self.n].copy()
        v = self.data.qvel[: self.n].copy()
        feedback = self.Kp @ (q_goal - q) - self.Kd @ v
        if self.controller_mode == "pd":
            return feedback
        if self.controller_mode == "gravity_compensated_pd":
            return self.data.qfrc_bias[: self.n].copy() + feedback
        raise RuntimeError(f"Unexpected controller_mode {self.controller_mode!r}.")

    def step(self, q_goal: np.ndarray) -> None:
        tau = self.compute_control(q_goal)
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

    def rollout(self, x0: np.ndarray, q_goal: np.ndarray, T: Optional[float] = None) -> np.ndarray:
        x0 = self._validate_vector("x0", x0, 2 * self.n)
        q_goal = self._validate_vector("q_goal", q_goal, self.n)
        horizon = self.T if T is None else float(T)
        if horizon < 0.0:
            raise ValueError(f"T must be nonnegative; got {horizon}.")

        self.reset(x0[: self.n], x0[self.n :])
        end_time = self.data.time + horizon
        while self.data.time < end_time - 0.5 * self.model.opt.timestep:
            self.step(q_goal)
        return self.get_state()

    def render(self, filename: str, height: int = 480, width: int = 640) -> None:
        mujoco_gl = os.environ.get("MUJOCO_GL", "").lower()
        if mujoco_gl not in {"egl", "osmesa", "glfw"}:
            print(
                "Warning: skipping MuJoCo rendering because MUJOCO_GL is not "
                "set to egl, osmesa, or glfw."
            )
            return
        try:
            with mujoco.Renderer(self.model, height=height, width=width) as renderer:
                renderer.update_scene(self.data, camera="fixed")
                image = renderer.render()
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(output_path, image)
        except Exception as exc:  # Rendering is optional and often environment-dependent.
            print(f"Warning: MuJoCo rendering failed for {filename!r}: {exc}")

    def print_dynamics_debug(self) -> None:
        mass_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
        print("M(q) =")
        print(mass_matrix[: self.n, : self.n])
        print("qfrc_bias[:n] =")
        print(self.data.qfrc_bias[: self.n])
