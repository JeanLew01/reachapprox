"""Reachable-set estimators built from endpoint samples (planar experiments)."""

from __future__ import annotations

from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import MultiPoint, Polygon


CHRISTOFFEL_DEGREE = 6
CHRISTOFFEL_REGULARIZATION = 1e-6


def convex_hull_polygon(points: np.ndarray) -> Polygon:
    """Convex hull of the endpoint samples as a shapely polygon."""
    hull = MultiPoint(points).convex_hull
    if hull.geom_type == "Polygon":
        return hull
    return hull.buffer(1e-10)


def make_grid(reference_points: np.ndarray, samples: np.ndarray, resolution: int) -> np.ndarray:
    """Square evaluation grid over the reference and sample points, with 8% padding."""
    all_points = np.vstack((reference_points, samples))
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    padding = 0.08 * np.maximum(upper - lower, 1e-6)
    xs = np.linspace(lower[0] - padding[0], upper[0] + padding[0], resolution)
    ys = np.linspace(lower[1] - padding[1], upper[1] + padding[1], resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def _ensure_nonempty(mask: np.ndarray, samples: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        distances, indices = cKDTree(grid_points).query(samples, k=1)
        mask[int(indices[int(np.argmin(distances))])] = True
    return mask


def convex_hull_mask(samples: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    """Grid points inside the convex hull of the samples."""
    hull = ConvexHull(samples)
    mask = MplPath(samples[hull.vertices]).contains_points(grid_points, radius=1e-12)
    return _ensure_nonempty(mask, samples, grid_points)


def monomial_powers(degree: int) -> list[tuple[int, int]]:
    return [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]


def polynomial_features(points: np.ndarray, powers: list[tuple[int, int]]) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    features = np.empty((points.shape[0], len(powers)), dtype=float)
    for k, (px, py) in enumerate(powers):
        features[:, k] = (x**px) * (y**py)
    return features


def christoffel_mask(samples: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    """Christoffel sublevel set {kappa_N <= max_i kappa_N(Y_i)} evaluated on a grid.

    kappa_N(y) = z_m(y)^T M^{-1} z_m(y) with degree-m monomials z_m and the
    ridge-regularized empirical moment matrix M (Appendix C.1, [18, Alg. 3]).
    """
    lower = grid_points.min(axis=0)
    upper = grid_points.max(axis=0)
    grid_center = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    scale[scale == 0.0] = 1.0

    powers = monomial_powers(CHRISTOFFEL_DEGREE)
    phi_samples = polynomial_features((samples - grid_center) / scale, powers)
    gram = (phi_samples.T @ phi_samples) / samples.shape[0]
    ridge = CHRISTOFFEL_REGULARIZATION * max(float(np.trace(gram)) / gram.shape[0], 1.0)
    inv_gram = np.linalg.pinv(gram + ridge * np.eye(gram.shape[0]), hermitian=True)

    k_samples = np.einsum("ij,jk,ik->i", phi_samples, inv_gram, phi_samples)
    threshold = float(np.max(k_samples)) * (1.0 + 1e-10)
    phi_grid = polynomial_features((grid_points - grid_center) / scale, powers)
    k_grid = np.einsum("ij,jk,ik->i", phi_grid, inv_gram, phi_grid)
    return _ensure_nonempty(k_grid <= threshold, samples, grid_points)
