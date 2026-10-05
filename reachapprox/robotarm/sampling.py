"""Initial-state sampling for the robot arm: uniform box and adversarial updates."""

from __future__ import annotations

import numpy as np

from .arm import propagate


RHO_Q = 0.1
RHO_V = 0.1
N_ADV = 1
ETA = 0.05
LAMBDA_REG = 1e-4


def sample_initial_box(rng: np.random.Generator, N: int, n: int) -> np.ndarray:
    """N uniform samples of x0 = [q0, v0] from [-rho_q, rho_q]^n x [-rho_v, rho_v]^n."""
    q0 = rng.uniform(-RHO_Q, RHO_Q, size=(N, n))
    v0 = rng.uniform(-RHO_V, RHO_V, size=(N, n))
    return np.hstack([q0, v0])


def project_initial_box(X: np.ndarray, n: int) -> np.ndarray:
    projected = np.asarray(X, dtype=float).copy()
    projected[:, :n] = np.clip(projected[:, :n], -RHO_Q, RHO_Q)
    projected[:, n:] = np.clip(projected[:, n:], -RHO_V, RHO_V)
    return projected


def adversarial_endpoints(
    rng: np.random.Generator,
    budget: int,
    n: int,
    T: float,
    n_adv: int = N_ADV,
    eta: float = ETA,
    lambda_reg: float = LAMBDA_REG,
) -> np.ndarray:
    """Algorithm 1 on the initial box, returning exactly `budget` endpoints.

    M = ceil(budget / (n_adv + 1)) uniform particles are propagated and moved
    n_adv times by a normalized step of length eta along Q (phi(T, x) - c),
    the novelty direction of L(x) = ||phi(T, x) - c||_Q^2 with the flow
    Jacobian replaced by the identity (the MuJoCo flow is not differentiated),
    then projected back onto the box.
    """
    if budget == 1:
        return propagate(sample_initial_box(rng, 1, n), n, T)

    X = sample_initial_box(rng, int(np.ceil(budget / (n_adv + 1))), n)
    Y = propagate(X, n, T)
    batches = [Y]
    for _ in range(n_adv):
        Y_acc = np.vstack(batches)
        center = np.mean(Y_acc, axis=0)
        centered = Y_acc - center
        cov = centered.T @ centered / (Y_acc.shape[0] - 1) if Y_acc.shape[0] > 1 else np.zeros((2 * n, 2 * n))
        Q = np.linalg.inv(cov + lambda_reg * np.eye(2 * n))
        direction = 2.0 * (Y - center) @ Q.T
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        X = project_initial_box(X + eta * direction, n)
        Y = propagate(X, n, T)
        batches.append(Y)
    return np.vstack(batches)[:budget]
