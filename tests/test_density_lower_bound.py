import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import kstest
from scipy.spatial import cKDTree

from experiments.density_lower_bound import (
    SUPPORT_CENTER,
    distance_to_ellipse,
    ellipse_boundary,
    greedy_maximal_packing,
    radial_density,
    sample_disk_density,
)


def test_exact_beta_radial_sampling_and_disk_membership() -> None:
    for beta in (0, 2, 4):
        samples = sample_disk_density(np.random.default_rng(100 + beta), 50_000, beta)
        radii = np.linalg.norm(samples - SUPPORT_CENTER, axis=1)
        assert np.all(radii <= 1.0)
        assert abs(radii.mean() - 2.0 / (beta + 3.0)) < 0.01
        assert kstest(radii, beta_distribution(2.0, beta + 1.0).cdf).statistic < 0.01


def test_beta_zero_is_uniform_on_the_disk() -> None:
    centered = sample_disk_density(np.random.default_rng(7), 50_000, 0) - SUPPORT_CENTER
    assert abs(np.sum(centered * centered, axis=1).mean() - 0.5) < 0.01


def test_positive_powers_vanish_at_boundary() -> None:
    near_boundary = np.array([1.0 - 1e-3, 1.0 - 1e-6])
    for beta in (2, 4):
        values = radial_density(near_boundary, beta)
        assert values[1] < values[0]
        assert radial_density(np.array([1.0]), beta)[0] == 0.0


def test_greedy_packing_is_separated_and_maximal() -> None:
    samples = np.random.default_rng(4).normal(size=(500, 2))
    centers = greedy_maximal_packing(samples, 0.2)
    assert cKDTree(centers).query(centers, k=2)[0][:, 1].min() > 0.2
    assert cKDTree(centers).query(samples, k=1)[0].max() <= 0.2 + 1e-12


def test_distance_to_ellipse() -> None:
    boundary = ellipse_boundary(64)
    assert np.allclose(distance_to_ellipse(boundary * 0.999 + 0.001 * boundary.mean(axis=0)), 0.0)
    outward = boundary + np.array([0.0, 0.5]) * np.sign(boundary[:, 1:2] + 1e-12)
    top = np.argmax(boundary[:, 1])
    assert np.isclose(distance_to_ellipse(outward[top : top + 1])[0], 0.5, atol=1e-3)
