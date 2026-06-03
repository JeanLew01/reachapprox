"""Compute Hausdorff Distance to the convex-hull robot-arm endpoint estimator."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial import cKDTree


MAX_HULL_VERTICES_FOR_DISTANCE = 500
FRANK_WOLFE_MAX_ITER = 30
FRANK_WOLFE_CHUNK_SIZE = 128
FRANK_WOLFE_TOL = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Reference .npz file with XT.")
    parser.add_argument("--sample", required=True, help="Smaller .npz file with XT.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional path for a one-line text result.")
    return parser.parse_args()


def farthest_point_coreset(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points

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


def hausdorff_distance_to_convex_hull(
    XT_ref: np.ndarray,
    XT_sample: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Numerical directed Hausdorff Distance from reference endpoints to conv(XT_sample)."""
    vertices = farthest_point_coreset(XT_sample, MAX_HULL_VERTICES_FOR_DISTANCE, rng)
    if vertices.shape[0] == 1:
        return float(np.linalg.norm(XT_ref - vertices[0], axis=1).max())

    tree = cKDTree(vertices)
    max_dist = 0.0
    for start in range(0, XT_ref.shape[0], FRANK_WOLFE_CHUNK_SIZE):
        Y = XT_ref[start : start + FRANK_WOLFE_CHUNK_SIZE]
        nearest = tree.query(Y, k=1, workers=-1)[1]
        Z = vertices[nearest].copy()

        for _ in range(FRANK_WOLFE_MAX_ITER):
            grad = Z - Y
            idx = np.argmin(grad @ vertices.T, axis=1)
            S = vertices[idx]
            direction = S - Z
            denom = np.sum(direction * direction, axis=1)
            active = denom > 1e-15
            step_size = np.zeros(Y.shape[0], dtype=float)
            step_size[active] = np.clip(
                -np.sum((Z[active] - Y[active]) * direction[active], axis=1) / denom[active],
                0.0,
                1.0,
            )
            Z += step_size[:, None] * direction
            if np.max(step_size * np.sqrt(np.maximum(denom, 0.0))) < FRANK_WOLFE_TOL:
                break

        max_dist = max(max_dist, float(np.linalg.norm(Y - Z, axis=1).max()))

    return max_dist


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
