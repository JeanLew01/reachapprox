"""Analytical flow utilities for dx/dt=x^2, dy/dt=0."""

from __future__ import annotations

import numpy as np


def flow(points: np.ndarray, T: float) -> np.ndarray:
    """Analytical flow phi(T, (x,y)) = (x/(1-Tx), y)."""
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    denom = 1.0 - T * x
    if np.any(denom <= 0.0):
        raise ValueError("Flow is singular for at least one point.")
    out = points.copy()
    out[:, 0] = x / denom
    return out


def flow_jacobian(points: np.ndarray, T: float) -> np.ndarray:
    """Return D_X phi(T, X) for each point, shape (n, 2, 2)."""
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    diag_x = 1.0 / (1.0 - T * x) ** 2
    jac = np.zeros((points.shape[0], 2, 2), dtype=float)
    jac[:, 0, 0] = diag_x
    jac[:, 1, 1] = 1.0
    return jac
