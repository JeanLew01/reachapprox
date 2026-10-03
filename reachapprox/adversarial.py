"""Adversarial sampling (Algorithm 1, after Lew and Pavone [19]) for x' = x^2, y' = 0."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .flows import quadratic_flow, quadratic_flow_jacobian
from .geometry import project_to_set, sample_uniform_polygon


ETA = 0.018
LAMBDA_REG = 1e-4


def endpoint_center_and_Q(endpoints: np.ndarray, lambda_reg: float = LAMBDA_REG) -> tuple[np.ndarray, np.ndarray]:
    """Mean c and regularized inverse covariance Q = (Cov + lambda I)^{-1} of the endpoint cloud."""
    n, dim = endpoints.shape
    c = np.sum(endpoints, axis=0) / n
    centered = endpoints - c
    cov = (centered.T @ centered) / (n - 1) if n > 1 else np.zeros((dim, dim), dtype=float)
    return c, np.linalg.inv(cov + lambda_reg * np.eye(dim))


def novelty_gradient(points: np.ndarray, T: float, c: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Gradient of L(x) = ||phi(T, x) - c||_Q^2 with respect to the initial state x."""
    qdiff = (quadratic_flow(points, T) - c) @ Q.T
    jac = quadratic_flow_jacobian(points, T)
    return 2.0 * np.einsum("nij,nj->ni", np.swapaxes(jac, 1, 2), qdiff)


def adversarial_endpoints(
    rng: np.random.Generator,
    geom: Polygon,
    T: float,
    budget: int,
    n_adv: int,
    eta: float = ETA,
) -> np.ndarray:
    """Return exactly `budget` endpoints from n_adv projected gradient-ascent updates.

    M = ceil(budget / (n_adv + 1)) uniform particles are propagated, then moved
    n_adv times along the novelty gradient computed from the accumulated
    endpoint cloud.  n_adv = 0 is i.i.d. uniform sampling.
    """
    particles = sample_uniform_polygon(rng, geom, int(np.ceil(budget / (n_adv + 1))))
    batches = [quadratic_flow(particles, T)]
    for _ in range(n_adv):
        c, Q = endpoint_center_and_Q(np.vstack(batches))
        particles = project_to_set(particles + eta * novelty_gradient(particles, T, c, Q), geom)
        batches.append(quadratic_flow(particles, T))
    return np.vstack(batches)[:budget]
