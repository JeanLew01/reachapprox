import numpy as np

from reachapprox.robotarm.metrics import convex_hull_directed_hausdorff, point_cloud_directed_hausdorff


def unit_hypercube(dim: int) -> np.ndarray:
    return np.array(np.meshgrid(*([[0.0, 1.0]] * dim))).T.reshape(-1, dim)


def test_point_cloud_metric() -> None:
    samples = np.array([[0.0, 0.0], [1.0, 0.0]])
    ref = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 2.0]])
    assert np.isclose(point_cloud_directed_hausdorff(ref, samples), 2.0)


def test_hull_metric_is_zero_for_interior_points() -> None:
    ref = np.array([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4], [0.7, 0.2, 0.8, 0.3]])
    assert convex_hull_directed_hausdorff(ref, unit_hypercube(4), np.random.default_rng(0)) == 0.0


def test_hull_metric_handles_arbitrary_convex_combinations() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(30, 6))
    ref = rng.dirichlet(np.ones(samples.shape[0]), size=5) @ samples
    assert convex_hull_directed_hausdorff(ref, samples, rng) == 0.0


def test_hull_metric_is_positive_for_exterior_points() -> None:
    ref = np.array([[1.2, 0.5, 0.5, 0.5]])
    assert np.isclose(convex_hull_directed_hausdorff(ref, unit_hypercube(4), np.random.default_rng(0)), 0.2)
