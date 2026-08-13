from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import kstest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.densitylower.density_lower_bound_experiment import (
    ESTIMATORS,
    ExperimentConfig,
    assert_complete_results,
    greedy_maximal_packing,
    linear_flow,
    radial_density,
    sample_disk_density,
    verify_maximal_packing,
)


def test_exact_beta_radial_sampling_and_disk_membership() -> None:
    n = 50_000
    for beta in (0, 2, 4):
        samples = sample_disk_density(np.random.default_rng(100 + beta), n, beta)
        radii = np.linalg.norm(samples - np.array([1.0, 0.0]), axis=1)
        assert np.all(radii <= 1.0)
        assert abs(radii.mean() - 2.0 / (beta + 3.0)) < 0.01
        statistic = kstest(radii, beta_distribution(2.0, beta + 1.0).cdf).statistic
        assert statistic < 0.01


def test_beta_zero_is_uniform_on_the_disk() -> None:
    samples = sample_disk_density(np.random.default_rng(7), 50_000, 0)
    centered = samples - np.array([1.0, 0.0])
    radius_squared = np.sum(centered * centered, axis=1)
    assert abs(radius_squared.mean() - 0.5) < 0.01


def test_positive_powers_vanish_at_boundary() -> None:
    near_boundary = np.array([1.0 - 1e-3, 1.0 - 1e-6])
    for beta in (2, 4):
        values = radial_density(near_boundary, beta)
        assert values[1] < values[0]
        assert radial_density(np.array([1.0]), beta)[0] == 0.0


def test_exact_linear_flow() -> None:
    config = ExperimentConfig()
    points = np.array([[1.0, 2.0], [-0.5, -3.0]])
    endpoints = linear_flow(points, config)
    assert np.allclose(endpoints[:, 0], np.exp(1.0) * points[:, 0])
    assert np.array_equal(endpoints[:, 1], points[:, 1])


def test_greedy_packing_is_separated_and_maximal() -> None:
    samples = np.random.default_rng(4).normal(size=(500, 2))
    centers = greedy_maximal_packing(samples, 0.2)
    verify_maximal_packing(samples, centers, 0.2)


def test_result_completeness_assertion() -> None:
    config = ExperimentConfig(betas=(0, 2), sample_budgets=(10,), n_seeds=2)
    rows = [
        {"beta": beta, "N": 10, "seed_index": seed, "estimator": estimator}
        for beta in config.betas
        for seed in range(config.n_seeds)
        for estimator in ESTIMATORS
    ]
    assert_complete_results(rows, config)
