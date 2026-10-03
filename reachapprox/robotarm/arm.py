"""Vertical planar serial n-link arm in MuJoCo under inverse-dynamics tracking control.

Dynamics:   M(q) v' + C(q, v) v + g(q) = tau,       state x = [q, v] in R^{2n}.
Reference:  q_d,i(t) = q_c,i + A sin(omega t + phi_i).
Controller: tau = M(q) q_d''(t) + C(q, v) v + g(q) - Kp e - Kd e',   e = q - q_d(t),
clipped to [-tau_limit, tau_limit].  The gains are deliberately weak (Appendix C.1).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os

import mujoco
import numpy as np


LINK_COUNTS = (2, 3, 4)
LINK_LENGTH = 0.5
TIMESTEP = 2e-3
KP = 0.01
KD = 0.005
TAU_LIMIT = 100.0
REFERENCE_AMPLITUDE = 0.08
REFERENCE_OMEGA = 0.5


def make_arm_xml(n: int, link_length: float = LINK_LENGTH, timestep: float = TIMESTEP) -> str:
    """MJCF model: chain in the xz-plane, hinges about the y-axis, gravity enabled."""
    body_xml = ""
    indent = "    "
    for i in range(n):
        body_xml += f'{indent}<body name="link{i + 1}" pos="{0.0 if i == 0 else link_length:.8g} 0 0">\n'
        body_xml += (
            f'{indent}  <joint name="joint{i + 1}" type="hinge" axis="0 1 0" '
            'damping="0.2" armature="0.01" limited="false"/>\n'
        )
        body_xml += (
            f'{indent}  <geom name="link{i + 1}_geom" type="capsule" '
            f'fromto="0 0 0 {link_length:.8g} 0 0" size="0.035" '
            'density="1000" contype="0" conaffinity="0"/>\n'
        )
        indent += "  "
    for _ in range(n):
        indent = indent[:-2]
        body_xml += f"{indent}</body>\n"

    actuator_xml = "".join(f'    <motor name="motor{i + 1}" joint="joint{i + 1}" gear="1"/>\n' for i in range(n))
    return (
        f'<mujoco model="vertical_planar_{n}_link_arm">\n'
        f'  <option timestep="{timestep:.8g}" gravity="0 0 -9.81"/>\n'
        "  <worldbody>\n"
        f"{body_xml}"
        "  </worldbody>\n"
        "  <actuator>\n"
        f"{actuator_xml}"
        "  </actuator>\n"
        "</mujoco>\n"
    )


def reference_trajectory(t: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q_d(t), q_d'(t), q_d''(t)."""
    q_c = np.linspace(0.25, 0.65, n)
    phase = REFERENCE_OMEGA * float(t) + np.linspace(0.0, np.pi / 3.0, n)
    amplitude = REFERENCE_AMPLITUDE * np.ones(n)
    return (
        q_c + amplitude * np.sin(phase),
        amplitude * REFERENCE_OMEGA * np.cos(phase),
        -amplitude * REFERENCE_OMEGA**2 * np.sin(phase),
    )


class NLinkArm:
    """Closed-loop simulator; `rollout` returns the state phi(T, x0)."""

    def __init__(self, n: int, kp: float = KP, kd: float = KD, tau_limit: float = TAU_LIMIT) -> None:
        if n not in LINK_COUNTS:
            raise ValueError(f"n must be one of {LINK_COUNTS}; got {n}.")
        self.n = n
        self.kp = kp
        self.kd = kd
        self.tau_limit = tau_limit
        self.model = mujoco.MjModel.from_xml_string(make_arm_xml(n))
        self.data = mujoco.MjData(self.model)
        self._inverse_data = mujoco.MjData(self.model)

    def control(self, t: float) -> np.ndarray:
        n = self.n
        q = self.data.qpos[:n].copy()
        v = self.data.qvel[:n].copy()
        qd, qd_dot, qd_ddot = reference_trajectory(t, n)

        # MuJoCo inverse dynamics: M(q) qd'' + C(q, v) v + g(q) (+ joint damping).
        self._inverse_data.qpos[:n] = q
        self._inverse_data.qvel[:n] = v
        self._inverse_data.qacc[:n] = qd_ddot
        mujoco.mj_inverse(self.model, self._inverse_data)
        tau_track = self._inverse_data.qfrc_inverse[:n].copy()

        tau = tau_track - self.kp * (q - qd) - self.kd * (v - qd_dot)
        return np.clip(tau, -self.tau_limit, self.tau_limit)

    def rollout(self, x0: np.ndarray, T: float) -> np.ndarray:
        n = self.n
        self.data.qpos[:n] = x0[:n]
        self.data.qvel[:n] = x0[n:]
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        while self.data.time < T - 0.5 * self.model.opt.timestep:
            self.data.ctrl[:] = self.control(self.data.time)
            mujoco.mj_step(self.model, self.data)
        return np.concatenate([self.data.qpos[:n], self.data.qvel[:n]])


def _propagate_serial(X0: np.ndarray, n: int, T: float) -> np.ndarray:
    arm = NLinkArm(n)
    return np.array([arm.rollout(x0, T) for x0 in X0]).reshape(X0.shape)


def propagate(X0: np.ndarray, n: int, T: float, workers: int | None = None) -> np.ndarray:
    """Endpoints phi(T, x0) for each row of X0.

    Rollouts are independent and deterministic, so large batches are split
    across `workers` processes (default: all CPUs) without changing the result.
    """
    if workers is None:
        workers = os.cpu_count() or 1
    if workers <= 1 or X0.shape[0] < 8 * workers:
        return _propagate_serial(X0, n, T)
    chunks = np.array_split(X0, workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.vstack(list(pool.map(_propagate_serial, chunks, [n] * workers, [T] * workers)))
