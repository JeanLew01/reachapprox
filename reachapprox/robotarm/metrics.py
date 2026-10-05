"""Directed Hausdorff errors for the robot-arm endpoint clouds.

The dimension-scaling experiment uses the distance from reference endpoints to
the convex hull of the samples,  max_{z in Y_ref} dist(z, conv(Y_1..Y_N)).  The
time sweep uses the point-cloud distance  max_{z in Y_ref} min_i ||z - Y_i||.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree


MAX_HULL_POINTS = 3000
NNLS_EQUALITY_WEIGHT = 1_000.0
NNLS_ZERO_TOL = 1e-8


def point_cloud_directed_hausdorff(ref_points: np.ndarray, sample_points: np.ndarray) -> float:
    """max_{z in ref_points} min_{y in sample_points} ||z - y||."""
    return float(cKDTree(sample_points).query(ref_points, k=1, workers=-1)[0].max())


def farthest_point_coreset(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point subset, used to bound the cost of hull projections."""
    if points.shape[0] <= max_points:
        return points
    selected = np.empty(max_points, dtype=int)
    selected[0] = int(rng.integers(0, points.shape[0]))
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)
    min_dist[selected[0]] = -np.inf
    for k in range(1, max_points):
        idx = int(np.argmax(min_dist))
        selected[k] = idx
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[idx], axis=1))
        min_dist[idx] = -np.inf
    return points[selected]


def _distance_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 1e-15:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(np.sum((points - a) * ab, axis=1) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _distances_to_convex_hull(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """dist(z, conv(vertices)) via nonnegative least squares over simplex weights."""
    if vertices.shape[0] == 1:
        return np.linalg.norm(points - vertices[0], axis=1)
    if vertices.shape[0] == 2:
        return _distance_to_segment(points, vertices[0], vertices[1])

    scale = max(1.0, float(np.linalg.norm(vertices, axis=1).max()))
    system = np.vstack([vertices.T / scale, NNLS_EQUALITY_WEIGHT * np.ones(vertices.shape[0])])
    rhs = np.empty(vertices.shape[1] + 1, dtype=float)
    rhs[-1] = NNLS_EQUALITY_WEIGHT
    maxiter = max(3 * vertices.shape[0], 1000)

    distances = np.empty(points.shape[0], dtype=float)
    for index, point in enumerate(points):
        rhs[:-1] = point / scale
        weights, _ = nnls(system, rhs, maxiter=maxiter)
        total = weights.sum()
        if total <= np.finfo(float).eps:
            distances[index] = np.linalg.norm(vertices - point, axis=1).min()
        else:
            distances[index] = np.linalg.norm(point - (weights / total) @ vertices)
    distances[distances < NNLS_ZERO_TOL] = 0.0
    return distances


def convex_hull_directed_hausdorff(
    ref_points: np.ndarray,
    sample_points: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """max_{z in ref_points} dist(z, conv(sample_points))."""
    vertices = farthest_point_coreset(sample_points, MAX_HULL_POINTS, rng)
    return float(np.max(_distances_to_convex_hull(ref_points, vertices)))
