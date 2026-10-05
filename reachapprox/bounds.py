"""Sample-complexity bounds for standard initial sets and one-sided Lipschitz fields.

Notation: n is the state dimension, mu the one-sided Lipschitz constant of the
field, T the horizon, R the volume-equivalent radius of the initial set S_0,
(kappa, lam) its standardness constants, D its diameter, rho the lower bound on
the sampling density relative to the uniform one, and delta the failure
probability.  The upper bounds control the inner error sup_{x in S_T} d(x, S_N)
of every estimator S_N that contains the endpoint samples.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def unit_ball_volume(n: int) -> float:
    return float(np.pi ** (n / 2.0) / gamma(n / 2.0 + 1.0))


def circular_cone_fraction(n: int, half_angle: float) -> float:
    """Normalized solid angle vartheta_n(alpha) of a circular cone of half-angle alpha in R^n."""
    cap = quad(lambda phi: np.sin(phi) ** (n - 2), 0.0, half_angle)[0]
    return float((n - 1) * unit_ball_volume(n - 1) / (n * unit_ball_volume(n)) * cap)


def spike_fraction(n: int) -> float:
    """kappa_n = vartheta_n(arctan(1/2)), the largest kappa covered by the spike lower bound."""
    return circular_cone_fraction(n, float(np.arctan(0.5)))


def upper_bound_samples(r, T, *, n: int, mu: float, R: float, kappa: float, lam: float,
                        rho: float = 1.0, delta: float = 0.05):
    """Upper bound for standard sets at every accuracy: N >= this gives inner error <= r w.p. 1 - delta.

    It depends on r and T only through the preimage scale min(e^{-mu T} r / 2, lam).
    """
    scale = np.minimum(0.5 * np.exp(-mu * np.asarray(T, dtype=float)) * r, lam)
    ratio = (R / scale) ** n
    return ratio / (rho * kappa) * (np.log(2.0**n * ratio / kappa) + np.log(1.0 / delta))


def convex_upper_bound_samples(r, T, *, n: int, mu: float, diameter: float, rho: float = 1.0,
                               delta: float = 0.05):
    """Upper bound for convex initial sets, which are ((R/D)^n, D)-standard and never saturate."""
    scale = np.minimum(0.5 * np.exp(-mu * np.asarray(T, dtype=float)) * r, diameter)
    return (diameter / scale) ** n / rho * (n * np.log(2.0 * diameter / scale) + np.log(1.0 / delta))


def required_samples(r, T, *, n: int, mu: float, R: float, kappa: float, lam: float, diameter: float,
                     convex: bool = False, rho: float = 1.0, delta: float = 0.05):
    """Smallest sample size certified by the upper bounds, over all three regimes.

    The standard-set bound, the convex bound if S_0 is convex, and one sample
    once r >= e^{mu T} diam(S_0).
    """
    r, T = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(T, dtype=float))
    needed = upper_bound_samples(r, T, n=n, mu=mu, R=R, kappa=kappa, lam=lam, rho=rho, delta=delta)
    if convex:
        needed = np.minimum(needed, convex_upper_bound_samples(r, T, n=n, mu=mu, diameter=diameter,
                                                               rho=rho, delta=delta))
    return np.where(r >= np.exp(mu * T) * diameter, 1.0, needed)


def certified_accuracy(N, T: float, *, resolution: int = 4000, **parameters):
    """Smallest accuracy r that `required_samples` certifies with N samples at horizon T."""
    diameter_T = np.exp(parameters["mu"] * T) * parameters["diameter"]
    radii = np.geomspace(1e-5 * diameter_T, diameter_T, resolution)
    needed = required_samples(radii, T, **parameters)
    certified = needed[None, :] <= np.atleast_1d(np.asarray(N, dtype=float))[:, None]
    accuracy = radii[np.argmax(certified, axis=1)]
    return accuracy if np.ndim(N) else float(accuracy[0])


def lower_bound_accuracy(N, T, *, n: int, mu: float, R: float, rho: float = 1.0, delta: float = 0.05,
                         kappa=None):
    """Minimax lower bound solved for r: with N samples no estimator achieves a smaller r on the family.

    Punctured balls if kappa is None, a ball with a spike of solid-angle
    fraction kappa otherwise.  NaN outside the admissible range
    r < e^{mu T} R / 16 (and kappa <= kappa_n for the spike).
    """
    N = np.asarray(N, dtype=float)
    scale = np.exp(mu * np.asarray(T, dtype=float)) * R
    log_term = np.log(1.0 / (2.0 * delta))
    if kappa is None:
        r = scale / 2.0 * (log_term / (2.0 * rho * N)) ** (1.0 / n)
        valid = r < scale / 16.0
    else:
        kappa = np.asarray(kappa, dtype=float)
        r = scale * 5.0 / 12.0 * (log_term / (2.0 * rho * kappa * N)) ** (1.0 / n)
        valid = (r < scale / 16.0) & (kappa <= spike_fraction(n))
    return np.where(valid, r, np.nan)


def lower_bound_samples(r, T, *, n: int, mu: float, R: float, rho: float = 1.0, delta: float = 0.05):
    """Minimax lower bound (punctured balls): with fewer samples every estimator fails on some instance."""
    scale = np.exp(mu * np.asarray(T, dtype=float)) * R
    needed = (scale / r) ** n / (2.0 ** (n + 1) * rho) * np.log(1.0 / (2.0 * delta))
    return np.where(r < scale / 16.0, needed, np.nan)
