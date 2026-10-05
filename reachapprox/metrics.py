"""Approximate Hausdorff distances and summary statistics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import QhullError, Voronoi, cKDTree
from shapely.geometry import Polygon

from .estimators import christoffel_mask, make_grid
from .geometry import boundary_points


CHRISTOFFEL_GRID_RESOLUTION = 90
HULL_CLOUD_SAMPLES = 30_000


def cloud_hausdorff(reference_points: np.ndarray, estimated_points: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two finite point clouds."""
    if estimated_points.shape[0] == 0:
        return float("inf")
    ref_to_est = cKDTree(estimated_points).query(reference_points, k=1)[0].max()
    est_to_ref = cKDTree(reference_points).query(estimated_points, k=1)[0].max()
    return float(max(ref_to_est, est_to_ref))


def cloud_inner_error(
    samples: np.ndarray,
    boundary: np.ndarray,
    contains: Callable[[np.ndarray], np.ndarray],
) -> float:
    """sup_{x in S} d(x, samples) for a compact planar set S, without gridding S.

    The distance to the nearest sample attains its maximum over S at a Voronoi
    vertex of the samples inside S or on the boundary of S.  `boundary` is a
    dense point set on the boundary of S and `contains` its membership test.
    When the samples lie in S, this is the Hausdorff distance between S and
    the sample cloud.
    """
    tree = cKDTree(samples)
    error = float(tree.query(boundary, k=1)[0].max())
    if samples.shape[0] < 3:
        return error
    try:
        vertices = Voronoi(samples).vertices
    except QhullError:
        return error
    in_box = np.all((vertices >= boundary.min(axis=0)) & (vertices <= boundary.max(axis=0)), axis=1)
    vertices = vertices[in_box]
    vertices = vertices[contains(vertices)]
    if vertices.shape[0]:
        error = max(error, float(tree.query(vertices, k=1)[0].max()))
    return error


def boundary_hausdorff(reference_boundary: np.ndarray, estimate: Polygon, n_points: int) -> float:
    """Symmetric Hausdorff distance between a reference boundary and a polygon boundary."""
    return cloud_hausdorff(reference_boundary, boundary_points(estimate, n_points))


def christoffel_hausdorff(reference_points: np.ndarray, endpoints: np.ndarray) -> float:
    """Hausdorff distance between a reference cloud and the Christoffel estimate."""
    grid_points = make_grid(reference_points, endpoints, CHRISTOFFEL_GRID_RESOLUTION)
    mask = christoffel_mask(endpoints, grid_points)
    # The threshold max_i kappa_N(Y_i) puts every sample in the sublevel set;
    # appending the samples keeps that inclusion in the discretized estimate.
    return cloud_hausdorff(reference_points, np.vstack((grid_points[mask], endpoints)))


def sample_from_polygon_fan(rng: np.random.Generator, hull: Polygon, n: int) -> np.ndarray:
    """Uniform samples from a convex polygon via a triangle fan around its vertex mean."""
    coords = np.asarray(hull.exterior.coords[:-1], dtype=float)
    if coords.shape[0] < 3:
        return np.repeat(coords[:1], n, axis=0)

    anchor = coords.mean(axis=0)
    starts = coords
    ends = np.roll(coords, -1, axis=0)
    areas = 0.5 * np.abs(np.cross(starts - anchor, ends - anchor))
    if float(np.sum(areas)) <= 0.0:
        raise ValueError("convex hull has zero area")

    tri = rng.choice(len(areas), size=n, p=areas / np.sum(areas))
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return anchor + u[:, None] * (starts[tri] - anchor) + v[:, None] * (ends[tri] - anchor)


def hull_hausdorff(rng: np.random.Generator, reference_points: np.ndarray, hull: Polygon) -> float:
    """Hausdorff distance between a reference cloud and a filled convex hull."""
    return cloud_hausdorff(reference_points, sample_from_polygon_fan(rng, hull, HULL_CLOUD_SAMPLES))
