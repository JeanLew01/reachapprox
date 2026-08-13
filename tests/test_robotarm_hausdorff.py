import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp.robotarm.fun.coverage import hausdorff_distance_to_convex_hull

def test_directed_hausdorff_to_convex_hull_is_zero_for_interior_points() -> None:
    sample_points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    ref_points = np.array([[0.25, 0.25], [0.5, 0.5]], dtype=float)
    rng = np.random.default_rng(0)
    error = hausdorff_distance_to_convex_hull(ref_points, sample_points, rng)
    assert error == 0.0

def test_directed_hausdorff_to_convex_hull_handles_high_dimensional_interior_points() -> None:
    sample_points = np.array(np.meshgrid(*([[0.0, 1.0]] * 4))).T.reshape(-1, 4)
    ref_points = np.array(
        [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4], [0.7, 0.2, 0.8, 0.3]],
        dtype=float,
    )

    rng = np.random.default_rng(0)
    error = hausdorff_distance_to_convex_hull(ref_points, sample_points, rng)

    assert error == 0.0

def test_directed_hausdorff_to_convex_hull_handles_arbitrary_convex_combinations() -> None:
    rng = np.random.default_rng(1)
    sample_points = rng.normal(size=(30, 6))
    weights = rng.dirichlet(np.ones(sample_points.shape[0]), size=5)
    ref_points = weights @ sample_points

    error = hausdorff_distance_to_convex_hull(ref_points, sample_points, rng)

    assert error == 0.0


def test_directed_hausdorff_to_convex_hull_is_positive_for_exterior_points() -> None:
    sample_points = np.array(np.meshgrid(*([[0.0, 1.0]] * 4))).T.reshape(-1, 4)
    ref_points = np.array([[1.2, 0.5, 0.5, 0.5]], dtype=float)

    rng = np.random.default_rng(0)
    error = hausdorff_distance_to_convex_hull(ref_points, sample_points, rng)

    assert np.isclose(error, 0.2)