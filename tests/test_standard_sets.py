import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

from reachapprox.flows import cubic_damping_flow, spiral_flow
from reachapprox.geometry import (
    contains_points,
    convex_polygon_standardness,
    dense_boundary,
    equilateral_triangle,
    isosceles_triangle,
    sample_uniform_set,
    square_array_set,
    square_set,
)
from reachapprox.metrics import cloud_inner_error


def test_spiral_flow_is_a_similarity_with_an_inverse() -> None:
    points = np.random.default_rng(0).normal(size=(20, 2))
    endpoints = spiral_flow(points, 0.7, rate=-1.0, omega=4.0)
    assert np.allclose(np.linalg.norm(endpoints, axis=1), np.exp(-0.7) * np.linalg.norm(points, axis=1))
    assert np.allclose(spiral_flow(endpoints, -0.7, rate=-1.0, omega=4.0), points)
    numeric = solve_ivp(lambda t, x: [-x[0] - 4.0 * x[1], 4.0 * x[0] - x[1]], (0.0, 0.7), points[0], rtol=1e-10,
                        atol=1e-12).y[:, -1]
    assert np.allclose(endpoints[0], numeric, atol=1e-7)


def test_cubic_damping_flow_solves_the_ode() -> None:
    points = np.array([[2.0, 1.0], [-0.5, 0.3]])
    endpoints = cubic_damping_flow(points, 1.5)
    for point, endpoint in zip(points, endpoints):
        numeric = solve_ivp(lambda t, x: -np.dot(x, x) * x, (0.0, 1.5), point, rtol=1e-10, atol=1e-12).y[:, -1]
        assert np.allclose(endpoint, numeric, atol=1e-7)
    assert np.allclose(cubic_damping_flow(endpoints, -1.5), points)


def test_new_initial_sets_have_area_pi() -> None:
    for geom in (isosceles_triangle(np.radians(7.5)), square_set(), square_array_set()):
        assert np.isclose(geom.area, np.pi)
        assert np.allclose([geom.centroid.x, geom.centroid.y], [2.0, 0.0], atol=1e-9)
    array = square_array_set(per_side=3, pitch_ratio=8.0)
    samples = sample_uniform_set(np.random.default_rng(1), array, 9_000)
    assert samples.shape == (9_000, 2)
    assert np.all(contains_points(array, samples))
    counts = [int(np.sum(contains_points(piece, samples))) for piece in array.geoms]
    assert min(counts) > 850 and max(counts) < 1_150


def test_polygon_standardness_constants() -> None:
    kappa, scale = convex_polygon_standardness(equilateral_triangle())
    assert np.isclose(kappa, 1.0 / 6.0) and np.isclose(scale, 0.778, atol=1e-3)
    kappa, scale = convex_polygon_standardness(square_set())
    assert np.isclose(kappa, 0.25) and np.isclose(scale, 0.5 * np.sqrt(np.pi))
    thin = isosceles_triangle(np.radians(15.0))
    kappa, scale = convex_polygon_standardness(thin)
    assert np.isclose(kappa, 15.0 / 360.0) and np.isclose(scale, 2.0 * thin.area / thin.length)


def test_cloud_inner_error_of_one_and_of_many_samples() -> None:
    square = square_set()
    boundary = dense_boundary(square, 1e-3)

    def contains(points: np.ndarray) -> np.ndarray:
        return contains_points(square, points)

    center = np.array([[2.0, 0.0]])
    assert np.isclose(cloud_inner_error(center, boundary, contains), np.sqrt(np.pi / 2.0))

    samples = sample_uniform_set(np.random.default_rng(2), square, 400)
    minx, miny, maxx, maxy = square.bounds
    xx, yy = np.meshgrid(np.linspace(minx, maxx, 700), np.linspace(miny, maxy, 700))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    brute_force = cKDTree(samples).query(grid, k=1)[0].max()
    exact = cloud_inner_error(samples, boundary, contains)
    assert abs(exact - brute_force) < 3e-3
