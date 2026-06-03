"""Sampling and adversarial updates for quadratic-dynamics experiments."""

from __future__ import annotations

from matplotlib.path import Path as MplPath
import numpy as np
from shapely.geometry import Polygon

from .quadratic_flow import flow, flow_jacobian
from .support_estimators import convex_hull_polygon


N_ADV = 9
ETA = 0.018
LAMBDA_REG = 1e-4


def polygon_path(geom: Polygon) -> MplPath:
    return MplPath(np.asarray(geom.exterior.coords))


def sample_uniform_polygon(rng: np.random.Generator, geom: Polygon, n: int) -> np.ndarray:
    minx, miny, maxx, maxy = geom.bounds
    area_box = (maxx - minx) * (maxy - miny)
    accept_rate = max(geom.area / area_box, 1e-3)
    batch = max(1024, int(np.ceil(1.4 * n / accept_rate)))
    path = polygon_path(geom)
    accepted: list[np.ndarray] = []
    count = 0

    while count < n:
        candidates = rng.uniform([minx, miny], [maxx, maxy], size=(batch, 2))
        mask = path.contains_points(candidates, radius=1e-12)
        chosen = candidates[mask]
        if chosen.size:
            accepted.append(chosen)
            count += chosen.shape[0]

    return np.vstack(accepted)[:n]


def project_to_polygon_boundary(points: np.ndarray, geom: Polygon, chunk_size: int = 512) -> np.ndarray:
    vertices = np.asarray(geom.exterior.coords[:-1], dtype=float)
    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    segs = ends - starts
    seg_norm2 = np.sum(segs * segs, axis=1)
    seg_norm2[seg_norm2 == 0.0] = 1.0

    out = np.empty_like(points)
    for start_idx in range(0, points.shape[0], chunk_size):
        chunk = points[start_idx : start_idx + chunk_size]
        diff = chunk[:, None, :] - starts[None, :, :]
        tau = np.einsum("nsi,si->ns", diff, segs) / seg_norm2[None, :]
        tau = np.clip(tau, 0.0, 1.0)
        candidates = starts[None, :, :] + tau[:, :, None] * segs[None, :, :]
        dist2 = np.sum((candidates - chunk[:, None, :]) ** 2, axis=2)
        nearest = np.argmin(dist2, axis=1)
        out[start_idx : start_idx + chunk.shape[0]] = candidates[np.arange(chunk.shape[0]), nearest]
    return out


def project_points_to_set(points: np.ndarray, geom: Polygon) -> np.ndarray:
    path = polygon_path(geom)
    inside = path.contains_points(points, radius=1e-12)
    projected = points.copy()
    if np.any(~inside):
        projected[~inside] = project_to_polygon_boundary(points[~inside], geom)
    return projected


def endpoint_center_and_Q(endpoints: np.ndarray, lambda_reg: float = LAMBDA_REG) -> tuple[np.ndarray, np.ndarray]:
    n = endpoints.shape[0]
    c = np.sum(endpoints, axis=0) / n
    centered = endpoints - c
    if n > 1:
        cov = (centered.T @ centered) / (n - 1)
    else:
        cov = np.zeros((2, 2), dtype=float)
    return c, np.linalg.inv(cov + lambda_reg * np.eye(2))


def adversarial_gradient(points: np.ndarray, T: float, c: np.ndarray, Q: np.ndarray) -> np.ndarray:
    endpoints = flow(points, T)
    diff = endpoints - c
    qdiff = diff @ Q.T
    jac = flow_jacobian(points, T)
    return 2.0 * np.einsum("nij,nj->ni", np.swapaxes(jac, 1, 2), qdiff)


def run_uniform_sampling(
    rng: np.random.Generator,
    geom: Polygon,
    T: float,
    n_samples: int,
) -> tuple[np.ndarray, Polygon]:
    x0 = sample_uniform_polygon(rng, geom, n_samples)
    endpoints = flow(x0, T)
    return endpoints, convex_hull_polygon(endpoints)


def run_adversarial_sampling(
    rng: np.random.Generator,
    geom: Polygon,
    T: float,
    total_budget: int,
    n_adv: int = N_ADV,
    eta: float = ETA,
) -> tuple[np.ndarray, Polygon]:
    if total_budget % (n_adv + 1) != 0:
        raise ValueError("total_budget must be divisible by n_adv + 1")

    m_particles = total_budget // (n_adv + 1)
    particles = sample_uniform_polygon(rng, geom, m_particles)
    endpoints = [flow(particles, T)]

    for _ in range(n_adv):
        accumulated = np.vstack(endpoints)
        c, Q = endpoint_center_and_Q(accumulated)
        grad = adversarial_gradient(particles, T, c, Q)
        particles = project_points_to_set(particles + eta * grad, geom)
        endpoints.append(flow(particles, T))

    endpoint_cloud = np.vstack(endpoints)
    return endpoint_cloud, convex_hull_polygon(endpoint_cloud)


def stable_seed_offset(set_name: str, method: str, t: float, budget: int) -> int:
    text = f"{set_name}:{method}:{t:.8f}:{budget}"
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))
