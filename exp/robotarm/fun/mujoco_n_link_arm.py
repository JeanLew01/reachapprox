"""MuJoCo vertical planar n-link arm dynamics for uncertainty propagation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import mujoco
import numpy as np


SUPPORTED_LINK_COUNTS = (2, 3, 4)
# Very weak tracking gains are intentional for this benchmark.  The experiment
# is about terminal uncertainty-set approximation, so the controller should move
# the arm along a slow reference without rapidly collapsing the sampled cloud.
DEFAULT_KP = 0.01
DEFAULT_KD = 0.005
DEFAULT_LAMBDA = 0.02
DEFAULT_GAMMA = 0.0
DEFAULT_SIGMA = 0.05
DEFAULT_TAU_LIMIT = 100.0
DEFAULT_REFERENCE_AMPLITUDE = 0.08
DEFAULT_REFERENCE_OMEGA = 0.5
TRACKING_CONTROLLER_NAME = "adaptive_inverse_dynamics_tracking"


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
             rgb1="0.08 0.16 0.30" rgb2="0.58 0.74 0.95"
             width="512" height="512"/>
    <material name="checker_mat" texture="checker" texrepeat="8 8" reflectance="0.1"/>
    <material name="link_mat" rgba="0.10 0.35 0.85 1"/>
  </asset>
"""

    worldbody_xml = """
  <worldbody>
    <light name="key_light" pos="0 -3 3" dir="0 1 -1" diffuse="1 1 1" ambient="0.45 0.45 0.45"/>
    <camera name="fixed" pos="0.55 -3.8 0.15" xyaxes="1 0 0 0 0 1" fovy="48"/>
    <geom name="ground" type="plane" pos="0 0 -1.2" size="3 3 0.01" material="checker_mat"/>
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


def reference_parameters(n: int) -> dict[str, np.ndarray | float]:
    """Return the slowly varying reference trajectory parameters.

    The low frequency is deliberate: this benchmark studies uncertainty-set
    propagation, so the reference should move the arm without immediately
    collapsing all sampled trajectories onto a static set point.
    """

    _validate_n(n)
    return {
        "q_c": np.linspace(0.25, 0.65, n),
        "A": DEFAULT_REFERENCE_AMPLITUDE * np.ones(n),
        "omega": DEFAULT_REFERENCE_OMEGA,
        "phi": np.linspace(0.0, np.pi / 3.0, n),
    }


def reference_trajectory(t: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q_d(t), qdot_d(t), and qddot_d(t) for the n-link arm."""

    params = reference_parameters(n)
    q_c = np.asarray(params["q_c"], dtype=float)
    amplitude = np.asarray(params["A"], dtype=float)
    omega = float(params["omega"])
    phi = np.asarray(params["phi"], dtype=float)
    phase = omega * float(t) + phi
    qd = q_c + amplitude * np.sin(phase)
    qd_dot = amplitude * omega * np.cos(phase)
    qd_ddot = -amplitude * omega**2 * np.sin(phase)
    return qd, qd_dot, qd_ddot


def inverse_dynamics_torque(
    model: mujoco.MjModel,
    q: np.ndarray,
    qdot: np.ndarray,
    qddot_des: np.ndarray,
    n: int,
    data: Optional[mujoco.MjData] = None,
) -> np.ndarray:
    """Compute ID(q, qdot, qddot_des) with MuJoCo inverse dynamics."""

    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    qddot_des = np.asarray(qddot_des, dtype=float)
    if q.shape != (n,) or qdot.shape != (n,) or qddot_des.shape != (n,):
        raise ValueError("q, qdot, and qddot_des must all have shape (n,).")

    tmp = mujoco.MjData(model) if data is None else data
    tmp.qpos[:n] = q
    tmp.qvel[:n] = qdot
    tmp.qacc[:n] = qddot_des
    mujoco.mj_inverse(model, tmp)
    return tmp.qfrc_inverse[:n].copy()


class AdaptiveIDTrackingController:
    """Adaptive inverse-dynamics trajectory tracking controller.

    Controller equations:

        e = q - q_d(t)
        edot = qdot - qdot_d(t)
        s = edot + Lambda e

        tau_ff = M(q) qddot_d(t) + C(q, qdot) qdot + g(q)

        d_hat_dot = Gamma s - sigma d_hat

        tau = tau_ff - Kp e - Kd edot - d_hat
    """

    def __init__(
        self,
        n: int,
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        lambda_gain: float = DEFAULT_LAMBDA,
        gamma: float = DEFAULT_GAMMA,
        sigma: float = DEFAULT_SIGMA,
        tau_limit: float = DEFAULT_TAU_LIMIT,
    ) -> None:
        _validate_n(n)
        if tau_limit <= 0.0:
            raise ValueError(f"tau_limit must be positive; got {tau_limit}.")
        self.n = n
        self.Kp = kp * np.eye(n)
        self.Kd = kd * np.eye(n)
        self.Lambda = lambda_gain * np.eye(n)
        self.Gamma = gamma * np.eye(n)
        self.sigma = float(sigma)
        self.tau_limit = float(tau_limit)
        self.d_hat = np.zeros(n)
        self._inverse_data: Optional[mujoco.MjData] = None

    def reset(self) -> None:
        self.d_hat = np.zeros(self.n)

    def compute_tau(self, model: mujoco.MjModel, data: mujoco.MjData, t: float, dt: float) -> np.ndarray:
        if self._inverse_data is None:
            self._inverse_data = mujoco.MjData(model)
        q = data.qpos[: self.n].copy()
        qdot = data.qvel[: self.n].copy()
        qd, qd_dot, qd_ddot = reference_trajectory(t, self.n)
        e = q - qd
        edot = qdot - qd_dot
        s = edot + self.Lambda @ e

        tau_ff = inverse_dynamics_torque(model, q, qdot, qd_ddot, self.n, self._inverse_data)
        self.d_hat += dt * (self.Gamma @ s - self.sigma * self.d_hat)
        tau = tau_ff - self.Kp @ e - self.Kd @ edot - self.d_hat
        return np.clip(tau, -self.tau_limit, self.tau_limit)

    def description(self) -> str:
        return (
            f"{TRACKING_CONTROLLER_NAME}(kp={self.Kp[0, 0]:.6g}, kd={self.Kd[0, 0]:.6g}, "
            f"lambda={self.Lambda[0, 0]:.6g}, gamma={self.Gamma[0, 0]:.6g}, "
            f"sigma={self.sigma:.6g}, tau_limit={self.tau_limit:.6g})"
        )


class MuJoCoNLinkArm:
    """Closed-loop MuJoCo simulator for a vertical planar serial n-link arm."""

    def __init__(
        self,
        n: int,
        link_length: float = 0.5,
        timestep: float = 0.002,
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        T: float = 2.0,
        controller_mode: str = TRACKING_CONTROLLER_NAME,
        lambda_gain: float = DEFAULT_LAMBDA,
        gamma: float = DEFAULT_GAMMA,
        sigma: float = DEFAULT_SIGMA,
        tau_limit: float = DEFAULT_TAU_LIMIT,
    ) -> None:
        _validate_n(n)
        if controller_mode != TRACKING_CONTROLLER_NAME:
            raise ValueError(
                f"controller_mode must be {TRACKING_CONTROLLER_NAME!r}; got {controller_mode!r}."
            )
        if T <= 0.0:
            raise ValueError(f"T must be positive; got {T}.")

        self.n = n
        self.link_length = link_length
        self.timestep = timestep
        self.T = T
        self.controller_mode = controller_mode

        xml = make_n_link_arm_xml(n=n, link_length=link_length, timestep=timestep)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.controller = AdaptiveIDTrackingController(
            n=n,
            kp=kp,
            kd=kd,
            lambda_gain=lambda_gain,
            gamma=gamma,
            sigma=sigma,
            tau_limit=tau_limit,
        )

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
        self.data.time = 0.0
        self.controller.reset()
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> np.ndarray:
        q = self.data.qpos[: self.n].copy()
        v = self.data.qvel[: self.n].copy()
        return np.concatenate([q, v])

    def compute_control(self, t: Optional[float] = None) -> np.ndarray:
        sim_time = self.data.time if t is None else float(t)
        return self.controller.compute_tau(self.model, self.data, sim_time, self.model.opt.timestep)

    def step(self) -> None:
        tau = self.compute_control()
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

    def rollout(self, x0: np.ndarray, q_goal: Optional[np.ndarray] = None, T: Optional[float] = None) -> np.ndarray:
        x0 = self._validate_vector("x0", x0, 2 * self.n)
        horizon = self.T if T is None else float(T)
        if horizon < 0.0:
            raise ValueError(f"T must be nonnegative; got {horizon}.")
        if q_goal is not None:
            self._validate_vector("q_goal", q_goal, self.n)

        self.reset(x0[: self.n], x0[self.n :])
        end_time = self.data.time + horizon
        while self.data.time < end_time - 0.5 * self.model.opt.timestep:
            self.step()
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
