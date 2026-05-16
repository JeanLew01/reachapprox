"""Dynamics for y_dot = 0, x_dot = x^2.

The default state convention is z = [x, y]. Under this convention,

    dz/dt = [x^2, 0].

The exact flow is available while 1 - dt * x0 is nonzero:

    x(t + dt) = x(t) / (1 - dt * x(t))
    y(t + dt) = y(t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

StateOrder = Literal["xy", "yx"]


def _as_state(z: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    state = np.asarray(z, dtype=float)
    if state.shape != (2,):
        raise ValueError(f"state must have shape (2,), got {state.shape}")
    return state


def f_continuous(
    z: np.ndarray | list[float] | tuple[float, float],
    order: StateOrder = "xy",
) -> np.ndarray:
    """Return the continuous-time derivative.

    Args:
        z: State vector. By default z = [x, y].
        order: Use "xy" for z = [x, y], or "yx" for z = [y, x].
    """
    state = _as_state(z)

    if order == "xy":
        x = state[0]
        return np.array([x**2, 0.0], dtype=float)
    if order == "yx":
        x = state[1]
        return np.array([0.0, x**2], dtype=float)

    raise ValueError(f"unknown state order {order!r}; expected 'xy' or 'yx'")


def jacobian(
    z: np.ndarray | list[float] | tuple[float, float],
    order: StateOrder = "xy",
) -> np.ndarray:
    """Return df/dz for the continuous-time dynamics."""
    state = _as_state(z)

    if order == "xy":
        x = state[0]
        return np.array([[2.0 * x, 0.0], [0.0, 0.0]], dtype=float)
    if order == "yx":
        x = state[1]
        return np.array([[0.0, 0.0], [0.0, 2.0 * x]], dtype=float)

    raise ValueError(f"unknown state order {order!r}; expected 'xy' or 'yx'")


@dataclass(frozen=True)
class QuadraticDynamics:
    """Small helper model for reachability experiments."""

    dt: float = 0.01
    order: StateOrder = "xy"

    n_x: int = 2
    y_dim: int = 2
    n_u: int = 0
    u_dim: int = 0

    def f_continuous(self, z: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
        return f_continuous(z, self.order)

    def jacobian(self, z: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
        return jacobian(z, self.order)

    def euler_step(
        self,
        z: np.ndarray | list[float] | tuple[float, float],
        dt: float | None = None,
    ) -> np.ndarray:
        h = self.dt if dt is None else float(dt)
        state = _as_state(z)
        return state + h * self.f_continuous(state)

    def rk4_step(
        self,
        z: np.ndarray | list[float] | tuple[float, float],
        dt: float | None = None,
    ) -> np.ndarray:
        h = self.dt if dt is None else float(dt)
        state = _as_state(z)
        k1 = self.f_continuous(state)
        k2 = self.f_continuous(state + 0.5 * h * k1)
        k3 = self.f_continuous(state + 0.5 * h * k2)
        k4 = self.f_continuous(state + h * k3)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def exact_step(
        self,
        z: np.ndarray | list[float] | tuple[float, float],
        dt: float | None = None,
    ) -> np.ndarray:
        """Return the analytic flow map for one step."""
        h = self.dt if dt is None else float(dt)
        state = _as_state(z).copy()

        x_index = 0 if self.order == "xy" else 1
        denominator = 1.0 - h * state[x_index]
        if np.isclose(denominator, 0.0):
            raise ZeroDivisionError("exact flow is singular when 1 - dt * x0 = 0")

        state[x_index] = state[x_index] / denominator
        return state

    def predict_mean(self, z: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
        """Discrete one-step prediction using the exact flow."""
        return self.exact_step(z)

    def predict_mean_linearized(
        self,
        z: np.ndarray | list[float] | tuple[float, float],
    ) -> np.ndarray:
        """Return the continuous-time linearization df/dz."""
        return self.jacobian(z)

    def simulate(
        self,
        z0: np.ndarray | list[float] | tuple[float, float],
        steps: int,
        method: Literal["exact", "rk4", "euler"] = "exact",
    ) -> np.ndarray:
        """Simulate and return an array with shape (steps + 1, 2)."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")

        trajectory = np.empty((steps + 1, 2), dtype=float)
        trajectory[0] = _as_state(z0)

        if method == "exact":
            step_fn = self.exact_step
        elif method == "rk4":
            step_fn = self.rk4_step
        elif method == "euler":
            step_fn = self.euler_step
        else:
            raise ValueError(f"unknown method {method!r}; expected 'exact', 'rk4', or 'euler'")

        for k in range(steps):
            trajectory[k + 1] = step_fn(trajectory[k])

        return trajectory
