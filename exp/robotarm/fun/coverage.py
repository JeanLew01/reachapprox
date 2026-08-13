"""Compute Hausdorff Distance to the convex-hull robot-arm endpoint estimator."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import ConvexHull, QhullError


MAX_HULL_VERTICES_FOR_DISTANCE = 3000
NNLS_EQUALITY_WEIGHT = 1_000.0
NNLS_ZERO_TOL = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Reference .npz file with XT.")
    parser.add_argument("--sample", required=True, help="Smaller .npz file with XT.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional path for a one-line text result.")
    return parser.parse_args()


def _distance_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 1e-15:
        return np.linalg.norm(points - a, axis=1)
    ap = points - a
    t = np.clip(np.sum(ap * ab, axis=1) / denom, 0.0, 1.0)
    projection = a + t[:, None] * ab
    return np.linalg.norm(points - projection, axis=1)


def _distance_to_convex_polygon_2d(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    if vertices.shape[0] == 1:
        return np.linalg.norm(points - vertices[0], axis=1)
    if vertices.shape[0] == 2:
        return _distance_to_segment(points, vertices[0], vertices[1])

    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    if hull_vertices.shape[0] < 3:
        return _distance_to_segment(points, hull_vertices[0], hull_vertices[1])

    area_sign = np.sign(
        sum(
            hull_vertices[i, 0] * hull_vertices[(i + 1) % hull_vertices.shape[0], 1]
            - hull_vertices[(i + 1) % hull_vertices.shape[0], 0] * hull_vertices[i, 1]
            for i in range(hull_vertices.shape[0])
        )
    )
    if area_sign == 0:
        area_sign = 1.0

    distances = np.zeros(points.shape[0], dtype=float)
    for index, point in enumerate(points):
        inside = True
        for start, end in zip(hull_vertices, np.roll(hull_vertices, -1, axis=0)):
            edge = end - start
            offset = point - start
            cross = edge[0] * offset[1] - edge[1] * offset[0]
            if area_sign > 0 and cross < -1e-12:
                inside = False
                break
            if area_sign < 0 and cross > 1e-12:
                inside = False
                break
        if inside:
            continue

        edge_distances = []
        for start, end in zip(hull_vertices, np.roll(hull_vertices, -1, axis=0)):
            edge_distances.append(_distance_to_segment(point[None, :], start, end)[0])
        distances[index] = min(edge_distances)

    return distances


def farthest_point_coreset(
    points: np.ndarray,
    max_points: int,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Select a deterministic-size geometric coreset when the sample is huge."""
    if points.shape[0] <= max_points:
        return points

    rng = np.random.default_rng(0) if rng is None else rng
    selected = np.empty(max_points, dtype=int)
    selected[0] = int(rng.integers(0, points.shape[0]))
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)
    min_dist[selected[0]] = -np.inf

    for k in range(1, max_points):
        idx = int(np.argmax(min_dist))
        selected[k] = idx
        dist = np.linalg.norm(points - points[idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[idx] = -np.inf

    return points[selected]


def _projection_distances_to_convex_hull(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Compute dist(point, conv(vertices)) by NNLS over simplex weights."""
    if vertices.shape[0] == 1:
        return np.linalg.norm(points - vertices[0], axis=1)
    if vertices.shape[0] == 2:
        return _distance_to_segment(points, vertices[0], vertices[1])

    distances = np.empty(points.shape[0], dtype=float)
    scale = max(1.0, float(np.linalg.norm(vertices, axis=1).max()))
    equality_weight = NNLS_EQUALITY_WEIGHT
    augmented_vertices = np.vstack(
        [vertices.T / scale, equality_weight * np.ones(vertices.shape[0])]
    )
    rhs = np.empty(vertices.shape[1] + 1, dtype=float)
    rhs[-1] = equality_weight
    maxiter = max(3 * vertices.shape[0], 1000)

    for index, point in enumerate(points):
        rhs[:-1] = point / scale
        weights, _ = nnls(augmented_vertices, rhs, maxiter=maxiter)
        total_weight = weights.sum()
        if total_weight <= np.finfo(float).eps:
            distances[index] = np.linalg.norm(vertices - point, axis=1).min()
            continue

        projection = (weights / total_weight) @ vertices
        distances[index] = np.linalg.norm(point - projection)

    distances[distances < NNLS_ZERO_TOL] = 0.0
    return distances


def hausdorff_distance_to_convex_hull(
    XT_ref: np.ndarray,
    XT_sample: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Directed Hausdorff distance from reference points to conv(XT_sample)."""
    XT_ref = np.asarray(XT_ref, dtype=float)
    XT_sample = np.asarray(XT_sample, dtype=float)

    if XT_ref.ndim != 2 or XT_sample.ndim != 2:
        raise ValueError("XT_ref and XT_sample must be two-dimensional arrays.")
    if XT_ref.shape[1] != XT_sample.shape[1]:
        raise ValueError("Reference and sample point dimensions must match.")
    if XT_ref.shape[0] == 0:
        return 0.0
    if XT_sample.shape[0] == 0:
        raise ValueError("XT_sample must contain at least one point.")

    vertices = farthest_point_coreset(XT_sample, MAX_HULL_VERTICES_FOR_DISTANCE, rng)

    if vertices.shape[1] == 2:
        try:
            hull_vertices = vertices
            if vertices.shape[0] >= 3:
                hull = ConvexHull(vertices)
                hull_vertices = vertices[hull.vertices]
            distances = _distance_to_convex_polygon_2d(XT_ref, hull_vertices)
            return float(np.max(distances))
        except QhullError:
            pass

    distances = _projection_distances_to_convex_hull(XT_ref, vertices)
    return float(np.max(distances))


def main() -> None:
    args = parse_args()
    ref_data = np.load(args.ref, allow_pickle=False)
    sample_data = np.load(args.sample, allow_pickle=False)
    XT_ref = np.asarray(ref_data["XT"], dtype=float)
    XT_sample = np.asarray(sample_data["XT"], dtype=float)
    if XT_ref.ndim != 2 or XT_sample.ndim != 2:
        raise ValueError("XT arrays must be two-dimensional.")
    if XT_ref.shape[1] != XT_sample.shape[1]:
        raise ValueError(
            f"State dimensions differ: ref has {XT_ref.shape[1]}, sample has {XT_sample.shape[1]}."
        )
    n = int(ref_data["n"]) if "n" in ref_data else XT_ref.shape[1] // 2
    if n not in (2, 3, 4):
        raise ValueError(f"Expected n=2, n=3, or n=4; got n={n}.")

    rng = np.random.default_rng(args.seed)
    error = hausdorff_distance_to_convex_hull(XT_ref, XT_sample, rng)
    result = f"Hausdorff Distance = {error:.12g}"
    print(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(result + "\n")


if __name__ == "__main__":
    main()
