import numpy as np

from reachapprox.robotarm.metrics import (
    _distances_to_convex_hull, convex_hull_directed_hausdorff,
    convex_hull_projection_certificate, point_cloud_directed_hausdorff,
)


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


def test_projection_certificate_contains_known_box_distances() -> None:
    rng = np.random.default_rng(904)
    for dim in (4, 8):
        # Include interior, near-boundary, exterior and translated cases.
        offset = rng.normal(size=dim)
        points = np.vstack([rng.uniform(-0.5, 1.5, size=(16, dim)), np.full((1, dim), 0.5)])
        exact = np.linalg.norm(points - np.clip(points, 0.0, 1.0), axis=1)
        vertices = unit_hypercube(dim) + offset
        reported, lower, upper = convex_hull_projection_certificate(points + offset, vertices)
        assert np.all(lower <= exact + 1e-13)
        assert np.all(exact <= upper + 1e-13)
        assert np.max(upper - lower) < 1e-5
        np.testing.assert_array_equal(reported, _distances_to_convex_hull(points + offset, vertices))


def test_projection_certificate_exposes_a_poor_feasible_candidate(monkeypatch) -> None:
    import reachapprox.robotarm.metrics as metrics

    vertices = unit_hypercube(4)
    points = np.array([[1.2, 0.5, 0.5, 0.5]])
    monkeypatch.setattr(metrics, "_convex_hull_candidates", lambda points, vertices: np.zeros_like(points))
    reported, lower, upper = metrics.convex_hull_projection_certificate(points, vertices)
    assert lower[0] <= 0.2 <= upper[0]
    assert upper[0] - lower[0] > 1.0
    assert reported[0] > 1.0


def test_projection_certificate_includes_zero_threshold_error() -> None:
    vertices = unit_hypercube(4)
    points = np.array([[1.0 + 1e-9, 0.5, 0.5, 0.5]])
    reported, lower, upper = convex_hull_projection_certificate(points, vertices)
    assert reported[0] == 0.0
    assert lower[0] <= 1e-9 <= upper[0]


def test_analytic_point_and_segment_distances_are_not_thresholded() -> None:
    points = np.array([[0.0, 1e-9]])
    for vertices in (np.array([[0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])):
        reported, _, _ = convex_hull_projection_certificate(points, vertices)
        np.testing.assert_array_equal(reported, np.array([1e-9]))
        np.testing.assert_array_equal(_distances_to_convex_hull(points, vertices), reported)
