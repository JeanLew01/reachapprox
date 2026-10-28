import numpy as np
import pytest

from experiments.robotarm_dimension_scaling import (
    TRIALS_CSV, SeedCurve, cluster_bootstrap_means, loglog_slopes, mean_errors,
)
from reachapprox.utils import read_csv


def proportional_seed_records(method: str = "adversarial", shared: bool = True) -> list[dict]:
    """Each seed multiplies an exact N**(-0.5) curve by a different factor."""
    return [
        {"method": method, "n": 2, "N": budget,
         "seed": seed + budget if method == "adversarial" or not shared else seed,
         "error": factor / np.sqrt(budget)}
        for budget in (1, 10, 100)
        for seed, factor in ((1000, 0.25), (1100, 1.0), (1200, 4.0), (1300, 10.0))
    ]


@pytest.mark.parametrize("method", ["adversarial", "uniform"])
def test_shared_seed_multipliers_do_not_introduce_slope_uncertainty(method: str) -> None:
    # Resampling whole records changes the curve's amplitude, never its slope.
    joint = loglog_slopes(mean_errors(proportional_seed_records(method)))[0]
    np.testing.assert_allclose([joint["slope"], joint["ci95_low"], joint["ci95_high"]], -0.5, atol=1e-14)


def test_disjoint_uniform_seeds_retain_slope_uncertainty() -> None:
    independent = loglog_slopes(mean_errors(proportional_seed_records("uniform", shared=False)))[0]
    assert independent["ci95_high"] - independent["ci95_low"] > 0.1


def test_seed_alignment_and_intervals_are_invariant_to_csv_row_order() -> None:
    rows = proportional_seed_records()
    # Break exact proportionality so an incorrect pairing can affect the CI.
    rows[4]["error"] *= 2.0
    expected = loglog_slopes(mean_errors(rows))
    np.random.default_rng(123).shuffle(rows)
    curves = mean_errors(rows)
    np.testing.assert_allclose(curves[("adversarial", 2)].errors[0], [0.25, 1.0, 4.0, 10.0])
    assert loglog_slopes(curves) == expected


def test_incomplete_seed_records_are_rejected_even_with_equal_budget_counts() -> None:
    rows = proportional_seed_records()
    rows[-1]["seed"] += 10
    with pytest.raises(ValueError, match="Incomplete adversarial seed records"):
        mean_errors(rows)


def test_duplicate_seed_records_are_rejected() -> None:
    rows = proportional_seed_records()
    rows.append(dict(rows[-1]))
    with pytest.raises(ValueError, match="Duplicate seed record"):
        mean_errors(rows)


def test_incomplete_uniform_panel_shares_weights_and_normalizes_available_records() -> None:
    rows = [
        {"method": "uniform", "n": 2, "N": budget, "seed": seed, "error": error}
        for budget, observations in ((1, ((10, 1.0), (20, 3.0))),
                                     (10, ((10, 2.0), (20, 6.0), (30, 100.0))))
        for seed, error in observations
    ]
    curve = mean_errors(rows)[("uniform", 2)]

    class FixedMultiplicities:
        def multinomial(self, n, probabilities, size):
            assert n == 3
            return np.resize([[1, 0, 2], [0, 3, 0], [3, 0, 0]], (size, 3))

    means = cluster_bootstrap_means(curve, FixedMultiplicities(), 3)
    np.testing.assert_allclose(means, [[1.0, 202.0 / 3], [3.0, 6.0], [1.0, 2.0]])


def test_empty_budget_resamples_are_redrawn() -> None:
    curve = SeedCurve(np.array([1, 10]), (np.array([2.0]), np.array([3.0])),
                      (np.array([10]), np.array([20])))
    means = cluster_bootstrap_means(curve, np.random.default_rng(0), 100)
    # Some cluster draws omit one budget. Retained draws have both finite means.
    np.testing.assert_allclose(means, np.tile([2.0, 3.0], (100, 1)))


def test_missing_observed_budget_records_are_rejected() -> None:
    curve = SeedCurve(np.array([1, 10]), (np.array([]), np.array([3.0])),
                      (np.array([], dtype=int), np.array([20])))
    with pytest.raises(ValueError, match="Every budget must contain"):
        cluster_bootstrap_means(curve, np.random.default_rng(0), 100)


def test_saved_trials_keep_point_slopes_and_are_invariant_to_row_order() -> None:
    rows = read_csv(TRIALS_CSV)
    curves = mean_errors(rows)
    expected = loglog_slopes(curves)
    np.random.default_rng(123).shuffle(rows)
    assert loglog_slopes(mean_errors(rows)) == expected
    # Actual saved uniform RNG collisions must remain visible to resampling.
    for n in (2, 3, 4):
        records = curves[("uniform", n)].records
        assert sum(map(len, records)) == 240
        assert len(np.unique(np.concatenate(records))) == 179
    slopes = {
        ("adversarial", 2): -0.42159786240994734,
        ("adversarial", 3): -0.2866338513424669,
        ("adversarial", 4): -0.23364223098362583,
        ("uniform", 2): -0.33336606942723634,
        ("uniform", 3): -0.2534845908603147,
        ("uniform", 4): -0.21647159271594793,
    }
    for row in expected:
        assert row["slope"] == slopes[(row["method"], row["n"])]


def test_corrected_adversarial_intervals_remain_unchanged() -> None:
    expected = {
        2: (-0.4378624468026677, -0.40541197989326666),
        3: (-0.29407807733839963, -0.2788501058979993),
        4: (-0.24049467794359208, -0.22740758372347014),
    }
    for row in loglog_slopes(mean_errors(read_csv(TRIALS_CSV))):
        if row["method"] == "adversarial":
            np.testing.assert_allclose(
                [row["ci95_low"], row["ci95_high"]], expected[row["n"]], rtol=0, atol=1e-14
            )
