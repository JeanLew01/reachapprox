import numpy as np

from reachapprox.bounds import (
    certified_accuracy,
    circular_cone_fraction,
    convex_upper_bound_samples,
    lower_bound_accuracy,
    lower_bound_samples,
    required_samples,
    spike_fraction,
    upper_bound_samples,
)


SQUARE = {"n": 2, "mu": -1.0, "R": 1.0, "kappa": 0.25, "lam": 0.886, "diameter": 2.507, "delta": 0.05}


def test_circular_cone_fractions() -> None:
    assert np.isclose(circular_cone_fraction(2, 0.4), 0.4 / np.pi)
    assert np.isclose(circular_cone_fraction(3, 0.7), (1.0 - np.cos(0.7)) / 2.0)
    assert np.allclose([circular_cone_fraction(n, np.pi / 2.0) for n in (2, 3, 4, 5)], 0.5)
    assert np.allclose([spike_fraction(n) for n in (2, 3, 4)], [0.148, 0.053, 0.020], atol=5e-4)


def test_upper_bound_for_the_equilateral_triangle() -> None:
    r, T, mu = 0.1, 0.3, -1.0
    u = np.exp(2.0 * mu * T) / r**2
    bound = upper_bound_samples(r, T, n=2, mu=mu, R=1.0, kappa=1.0 / 6.0, lam=0.7776, delta=0.05)
    assert np.isclose(bound, 24.0 * u * (np.log(96.0 * u) + np.log(20.0)))


def test_upper_bound_depends_on_the_preimage_scale_only() -> None:
    parameters = {key: SQUARE[key] for key in ("n", "mu", "R", "kappa", "lam", "delta")}
    assert np.isclose(upper_bound_samples(0.05, 0.0, **parameters),
                      upper_bound_samples(0.05 * np.exp(-1.5), 1.5, **parameters))
    saturated = upper_bound_samples(np.array([2.0, 5.0, 50.0]), 0.0, **parameters)
    assert np.allclose(saturated, saturated[0])


def test_required_samples_has_three_regimes() -> None:
    radii = np.geomspace(1e-3, 10.0, 400)
    needed = required_samples(radii, 0.5, **SQUARE)
    assert np.all(np.diff(needed) <= 1e-9)
    assert np.all(needed[radii >= np.exp(-0.5) * SQUARE["diameter"]] == 1.0)
    convex = required_samples(radii, 0.5, convex=True, **SQUARE)
    assert np.all(convex <= needed)
    assert np.isclose(convex_upper_bound_samples(2.0 * 2.507, 0.0, n=2, mu=-1.0, diameter=2.507),
                      2.0 * np.log(2.0) + np.log(20.0))


def test_certified_accuracy_inverts_the_bound() -> None:
    for sample_size in (30.0, 1_000.0, 100_000.0):
        r = certified_accuracy(sample_size, 1.0, **SQUARE)
        assert required_samples(r, 1.0, **SQUARE) <= sample_size
        assert required_samples(0.98 * r, 1.0, **SQUARE) > sample_size
    assert np.isclose(certified_accuracy(1.0, 1.0, **SQUARE), np.exp(-1.0) * SQUARE["diameter"])


def test_lower_bounds_and_their_admissible_range() -> None:
    r = lower_bound_accuracy(1e4, 0.5, n=2, mu=-1.0, R=1.0)
    assert np.isclose(lower_bound_samples(r, 0.5, n=2, mu=-1.0, R=1.0), 1e4)
    assert np.isnan(lower_bound_accuracy(10.0, 0.0, n=2, mu=0.0, R=1.0))
    spike = lower_bound_accuracy(1e4, 0.0, n=2, mu=0.0, R=1.0, kappa=np.array([1.0 / 6.0, 1.0 / 24.0]))
    assert np.isnan(spike[0])
    assert np.isclose(spike[1], 5.0 / 12.0 * np.sqrt(np.log(10.0) * 24.0 / 2e4))
