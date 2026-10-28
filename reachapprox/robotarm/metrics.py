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


def _convex_hull_candidates(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Feasible projection candidates from normalized, equality-penalized NNLS.

    The finite equality penalty does not ensure an exact Euclidean projection.
    Normalization makes the candidate a convex combination (up to roundoff).
    """
    if vertices.shape[0] == 1:
        return np.broadcast_to(vertices[0], points.shape).copy()
    if vertices.shape[0] == 2:
        ab = vertices[1] - vertices[0]
        denom = np.dot(ab, ab)
        if denom <= 1e-15:
            return np.broadcast_to(vertices[0], points.shape).copy()
        t = np.clip(np.sum((points - vertices[0]) * ab, axis=1) / denom, 0.0, 1.0)
        return vertices[0] + t[:, None] * ab

    scale = max(1.0, float(np.linalg.norm(vertices, axis=1).max()))
    system = np.vstack([vertices.T / scale, NNLS_EQUALITY_WEIGHT * np.ones(vertices.shape[0])])
    rhs = np.empty(vertices.shape[1] + 1, dtype=float)
    rhs[-1] = NNLS_EQUALITY_WEIGHT
    maxiter = max(3 * vertices.shape[0], 1000)

    candidates = np.empty_like(points, dtype=float)
    for index, point in enumerate(points):
        rhs[:-1] = point / scale
        weights, _ = nnls(system, rhs, maxiter=maxiter)
        total = weights.sum()
        if total <= np.finfo(float).eps:
            candidates[index] = vertices[np.argmin(np.linalg.norm(vertices - point, axis=1))]
        else:
            candidates[index] = (weights / total) @ vertices
    return candidates


def _distances_to_convex_hull(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Approximate hull distances; see ``convex_hull_projection_certificate``.

    Before the zero threshold the feasible-candidate distances are upper bounds
    in exact arithmetic. The reference cloud is a separate approximation.
    """
    distances = np.linalg.norm(points - _convex_hull_candidates(points, vertices), axis=1)
    if vertices.shape[0] > 2:
        distances[distances < NNLS_ZERO_TOL] = 0.0
    return distances


def convex_hull_projection_certificate(
    points: np.ndarray, vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reported distances and numerical primal/dual distance bounds.

    For a feasible candidate y and z outside it, let u = (y-z)/||y-z||.
    Cauchy--Schwarz gives max(0, min_i u.T (v_i-z)) <= dist(z, conv(V))
    <= ||y-z||. The bound width equals the Frank--Wolfe gap divided by
    ||y-z|| until the lower bound reaches zero. No second optimization is needed.

    A conservative roundoff cushion is applied to both bounds. These are
    floating-point diagnostics, not interval-arithmetic certificates. The
    reported NNLS distances include the production 1e-8 zero threshold; the
    analytic one-/two-vertex branches retain their unthresholded distances.
    """
    candidates = _convex_hull_candidates(points, vertices)
    residuals = candidates - points
    raw = np.linalg.norm(residuals, axis=1)
    lower = np.zeros_like(raw)
    for index in np.flatnonzero(raw):
        direction = residuals[index] / raw[index]
        lower[index] = max(0.0, float(np.min((vertices - points[index]) @ direction)))
    magnitude = 1.0 + np.linalg.norm(points, axis=1) + np.max(np.linalg.norm(vertices, axis=1))
    cushion = 64 * np.finfo(float).eps * vertices.shape[1] * magnitude
    lower = np.maximum(0.0, lower - cushion)
    upper = raw + cushion
    distances = raw.copy()
    if vertices.shape[0] > 2:
        distances[distances < NNLS_ZERO_TOL] = 0.0
    return distances, lower, upper


def convex_hull_directed_hausdorff(
    ref_points: np.ndarray,
    sample_points: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """max_{z in ref_points} dist(z, conv(sample_points))."""
    vertices = farthest_point_coreset(sample_points, MAX_HULL_POINTS, rng)
    return float(np.max(_distances_to_convex_hull(ref_points, vertices)))
