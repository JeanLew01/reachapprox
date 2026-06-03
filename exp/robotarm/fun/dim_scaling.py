"""Shared helpers for robot-arm uncertainty-propagation experiments."""

from __future__ import annotations

import numpy as np

from .coverage import hausdorff_distance_to_convex_hull
from .mujoco_n_link_arm import (
    TRACKING_CONTROLLER_NAME,
    MuJoCoNLinkArm,
)


DEFAULT_LINK_COUNTS = (2, 3, 4)
DEFAULT_SAMPLE_BUDGETS = (1, 3, 10, 30, 100, 300, 1000, 3000)
DEFAULT_N_SEEDS = 20
DEFAULT_N_REF = 50_000
DEFAULT_COVERAGE_SUBSET = 2_000

T_HORIZON = 1.0
RHO_Q = 0.1
RHO_V = 0.1
CONTROLLER_MODE = TRACKING_CONTROLLER_NAME
REFERENCE_SEED = 202606
EXPERIMENT_SEED = 77531

METRIC_NAME = "Hausdorff Distance"
METRIC_IMPLEMENTATION = "directed_hausdorff_to_convex_hull"


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def controller_label(
    kp: float,
    kd: float,
    lambda_gain: float,
    gamma: float,
    sigma: float,
    tau_limit: float,
) -> str:
    return (
        f"{TRACKING_CONTROLLER_NAME}(kp={kp:.6g},kd={kd:.6g},lambda={lambda_gain:.6g},"
        f"gamma={gamma:.6g},sigma={sigma:.6g},tau_limit={tau_limit:.6g})"
    )


def sample_initial_box(
    rng: np.random.Generator,
    N: int,
    n: int,
    rho_q: float,
    rho_v: float,
) -> np.ndarray:
    q0 = rng.uniform(-rho_q, rho_q, size=(N, n))
    v0 = rng.uniform(-rho_v, rho_v, size=(N, n))
    return np.hstack([q0, v0])


def propagate_endpoints(
    X0: np.ndarray,
    n: int,
    T: float,
    kp: float,
    kd: float,
    lambda_gain: float,
    gamma: float,
    sigma: float,
    tau_limit: float,
) -> np.ndarray:
    arm = MuJoCoNLinkArm(
        n=n,
        T=T,
        kp=kp,
        kd=kd,
        lambda_gain=lambda_gain,
        gamma=gamma,
        sigma=sigma,
        tau_limit=tau_limit,
    )
    XT = np.empty_like(X0)
    for i, x0 in enumerate(X0):
        XT[i] = arm.rollout(x0, T=T)
    return XT


def directed_hausdorff_to_convex_hull(
    ref_points: np.ndarray,
    sample_points: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Compute max_z dist(z, conv(sample_points)).

    This is the convex-hull reachable-set estimator metric used by the robot-arm
    dimension-scaling experiments after the controller and flow are fixed.
    """

    ref_points = np.asarray(ref_points, dtype=float)
    sample_points = np.asarray(sample_points, dtype=float)
    if ref_points.ndim != 2 or sample_points.ndim != 2:
        raise ValueError("ref_points and sample_points must be two-dimensional.")
    if ref_points.shape[1] != sample_points.shape[1]:
        raise ValueError("Point dimensions must match.")
    return hausdorff_distance_to_convex_hull(ref_points, sample_points, rng)


def mean_and_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=1)
    if values.shape[1] == 1:
        return mean, mean, mean
    stderr = np.std(values, axis=1, ddof=1) / np.sqrt(values.shape[1])
    half_width = 1.96 * stderr
    return mean, np.maximum(mean - half_width, np.finfo(float).tiny), mean + half_width
