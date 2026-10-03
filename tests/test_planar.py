import numpy as np

from reachapprox.adversarial import adversarial_endpoints
from reachapprox.estimators import christoffel_mask, convex_hull_mask, make_grid
from reachapprox.flows import linear_flow, quadratic_flow, quadratic_flow_jacobian
from reachapprox.geometry import (
    equal_area_initial_sets,
    opened_triangle_reach,
    project_to_set,
    sample_uniform_polygon,
)


def test_quadratic_flow_matches_its_jacobian() -> None:
    points = np.array([[0.5, 1.0], [1.5, -2.0]])
    T, eps = 0.2, 1e-6
    numeric = (quadratic_flow(points + [eps, 0.0], T) - quadratic_flow(points - [eps, 0.0], T)) / (2 * eps)
    assert np.allclose(numeric[:, 0], quadratic_flow_jacobian(points, T)[:, 0, 0], rtol=1e-6)


def test_linear_flow_is_exponential_in_x_only() -> None:
    points = np.array([[1.0, 2.0], [-0.5, -3.0]])
    endpoints = linear_flow(points, 0.5, rate=2.0)
    assert np.allclose(endpoints[:, 0], np.e * points[:, 0])
    assert np.array_equal(endpoints[:, 1], points[:, 1])


def test_equal_area_initial_sets() -> None:
    for item in equal_area_initial_sets():
        assert np.isclose(item.geom.area, np.pi, rtol=1e-3)
        assert np.allclose([item.geom.centroid.x, item.geom.centroid.y], [2.0, 0.0], atol=1e-3)
    assert np.isclose(opened_triangle_reach(), 0.309, atol=1e-3)


def test_projection_lands_in_the_set() -> None:
    geom = equal_area_initial_sets()[2].geom
    points = np.random.default_rng(0).normal(loc=2.0, scale=3.0, size=(200, 2))
    projected = project_to_set(points, geom)
    assert np.all([geom.buffer(1e-9).contains(type(geom.centroid)(p)) for p in projected])


def test_zero_adversarial_updates_is_uniform_sampling() -> None:
    geom = equal_area_initial_sets()[0].geom
    adversarial = adversarial_endpoints(np.random.default_rng(3), geom, 0.1, 50, n_adv=0)
    uniform = quadratic_flow(sample_uniform_polygon(np.random.default_rng(3), geom, 50), 0.1)
    assert np.array_equal(adversarial, uniform)


def test_estimators_contain_the_samples_region() -> None:
    samples = np.random.default_rng(1).uniform(-1.0, 1.0, size=(400, 2))
    grid = make_grid(samples, samples, 60)
    inside_square = np.all(np.abs(grid) < 0.8, axis=1)
    assert np.all(convex_hull_mask(samples, grid)[inside_square])
    assert np.mean(christoffel_mask(samples, grid)[inside_square]) > 0.99
