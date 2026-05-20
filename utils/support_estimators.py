"""Support estimators and Hausdorff approximations."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon


CHRISTOFFEL_DEGREE = 6
CHRISTOFFEL_REGULARIZATION = 1e-6
CHRISTOFFEL_GRID_RESOLUTION = 90
HULL_CLOUD_SAMPLES = 30_000


def convex_hull_polygon(points: np.ndarray) -> Polygon:
    hull = MultiPoint(points).convex_hull
    if hull.geom_type == "Polygon":
        return hull
    return hull.buffer(1e-10)


def monomial_powers(degree: int) -> list[tuple[int, int]]:
    return [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]


def polynomial_features(points: np.ndarray, powers: list[tuple[int, int]]) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    features = np.empty((points.shape[0], len(powers)), dtype=float)
    for k, (px, py) in enumerate(powers):
        features[:, k] = (x**px) * (y**py)
    return features


def christoffel_estimator_mask(samples: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    lower = grid_points.min(axis=0)
    upper = grid_points.max(axis=0)
    grid_center = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    scale[scale == 0.0] = 1.0

    samples_scaled = (samples - grid_center) / scale
    grid_scaled = (grid_points - grid_center) / scale

    powers = monomial_powers(CHRISTOFFEL_DEGREE)
    phi_samples = polynomial_features(samples_scaled, powers)
    gram = (phi_samples.T @ phi_samples) / samples.shape[0]
    ridge = CHRISTOFFEL_REGULARIZATION * max(float(np.trace(gram)) / gram.shape[0], 1.0)
    gram = gram + ridge * np.eye(gram.shape[0])

    inv_gram = np.linalg.pinv(gram, hermitian=True)
    k_samples = np.einsum("ij,jk,ik->i", phi_samples, inv_gram, phi_samples)
    threshold = float(np.max(k_samples)) * (1.0 + 1e-10)

    phi_grid = polynomial_features(grid_scaled, powers)
    k_grid = np.einsum("ij,jk,ik->i", phi_grid, inv_gram, phi_grid)
    mask = k_grid <= threshold

    if not np.any(mask):
        distances, indices = cKDTree(grid_points).query(samples, k=1)
        nearest_sample = int(np.argmin(distances))
        mask[int(indices[nearest_sample])] = True

    return mask


def make_support_grid(reference_points: np.ndarray, samples: np.ndarray) -> np.ndarray:
    all_points = np.vstack((reference_points, samples))
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    span = upper - lower
    padding = 0.08 * np.maximum(span, 1e-6)
    lower = lower - padding
    upper = upper + padding

    xs = np.linspace(lower[0], upper[0], CHRISTOFFEL_GRID_RESOLUTION)
    ys = np.linspace(lower[1], upper[1], CHRISTOFFEL_GRID_RESOLUTION)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def approximate_support_hausdorff(reference_points: np.ndarray, estimated_points: np.ndarray) -> float:
    if estimated_points.shape[0] == 0:
        return float("inf")
    tree_ref = cKDTree(reference_points)
    tree_est = cKDTree(estimated_points)
    ref_to_est = tree_est.query(reference_points, k=1)[0].max()
    est_to_ref = tree_ref.query(estimated_points, k=1)[0].max()
    return float(max(ref_to_est, est_to_ref))


def christoffel_support_hausdorff(reference_points: np.ndarray, endpoint_samples: np.ndarray) -> float:
    grid_points = make_support_grid(reference_points, endpoint_samples)
    mask = christoffel_estimator_mask(endpoint_samples, grid_points)
    # tau_N = max_i kappa_N(y_i) includes samples in the continuous sublevel set.
    # Appending them here preserves that inclusion in the discrete Hausdorff cloud.
    estimated_points = np.vstack((grid_points[mask], endpoint_samples))
    return approximate_support_hausdorff(reference_points, estimated_points)


def sample_from_convex_hull(rng: np.random.Generator, hull: Polygon, n: int) -> np.ndarray:
    coords = np.asarray(hull.exterior.coords[:-1], dtype=float)
    if coords.shape[0] < 3:
        return np.repeat(coords[:1], n, axis=0)

    anchor = coords.mean(axis=0)
    starts = coords
    ends = np.roll(coords, -1, axis=0)
    cross = np.abs(np.cross(starts - anchor, ends - anchor))
    areas = 0.5 * cross
    if float(np.sum(areas)) <= 0.0:
        raise ValueError("convex hull has zero area")

    tri_indices = rng.choice(len(areas), size=n, p=areas / np.sum(areas))
    a = np.repeat(anchor[None, :], n, axis=0)
    b = starts[tri_indices]
    c = ends[tri_indices]
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return a + u[:, None] * (b - a) + v[:, None] * (c - a)


def approximate_hausdorff(
    rng: np.random.Generator,
    reference_points: np.ndarray,
    estimate_hull: Polygon,
    n_hull_samples: int = HULL_CLOUD_SAMPLES,
) -> float:
    estimate_points = sample_from_convex_hull(rng, estimate_hull, n_hull_samples)
    tree_ref = cKDTree(reference_points)
    tree_est = cKDTree(estimate_points)
    ref_to_est = tree_est.query(reference_points, k=1)[0].max()
    est_to_ref = tree_ref.query(estimate_points, k=1)[0].max()
    return float(max(ref_to_est, est_to_ref))


def approximate_boundary_hausdorff(reference_boundary: np.ndarray, estimate_hull: Polygon) -> float:
    from .quaddyn_geometry import boundary_points

    estimate_boundary = boundary_points(estimate_hull, 2_000)
    tree_ref = cKDTree(reference_boundary)
    tree_est = cKDTree(estimate_boundary)
    ref_to_est = tree_est.query(reference_boundary, k=1)[0].max()
    est_to_ref = tree_ref.query(estimate_boundary, k=1)[0].max()
    return float(max(ref_to_est, est_to_ref))
