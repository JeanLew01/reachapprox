"""Analytic flows of the planar autonomous systems used in the experiments."""

from __future__ import annotations

import numpy as np


def quadratic_flow(points: np.ndarray, T: float) -> np.ndarray:
    """Flow of x' = x^2, y' = 0: phi(T, (x, y)) = (x / (1 - T x), y)."""
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    denom = 1.0 - T * x
    if np.any(denom <= 0.0):
        raise ValueError("Flow is singular for at least one point.")
    out = points.copy()
    out[:, 0] = x / denom
    return out


def quadratic_flow_jacobian(points: np.ndarray, T: float) -> np.ndarray:
    """Return D_x phi(T, x) of the quadratic flow for each point, shape (n, 2, 2)."""
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    jac = np.zeros((points.shape[0], 2, 2), dtype=float)
    jac[:, 0, 0] = 1.0 / (1.0 - T * x) ** 2
    jac[:, 1, 1] = 1.0
    return jac


def linear_flow(points: np.ndarray, T: float, rate: float = 1.0) -> np.ndarray:
    """Flow of x' = rate * x, y' = 0: phi(T, (x, y)) = (e^{rate T} x, y)."""
    out = np.asarray(points, dtype=float).copy()
    out[:, 0] *= np.exp(rate * T)
    return out
