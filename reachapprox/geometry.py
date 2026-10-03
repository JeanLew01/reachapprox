"""Initial sets, uniform sampling, and projection for the planar experiments.

All planar initial sets are centered at (2, 0).  The equal-area family of
Section 5.1 and Appendix C.3 (circle, opened triangle, triangle) has area pi.
"""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.path import Path as MplPath
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon


CENTER = np.array([2.0, 0.0])
DISK_RADIUS = 1.0
TARGET_AREA = np.pi
OPENING_RADIUS = 0.3
DISK_RESOLUTION = 512
BUFFER_RESOLUTION = 96


@dataclass(frozen=True)
class InitialSet:
    name: str
    label: str
    geom: Polygon


def disk_set() -> Polygon:
    return Point(CENTER).buffer(DISK_RADIUS, quad_segs=DISK_RESOLUTION)


def equilateral_triangle() -> Polygon:
    """Equilateral triangle with area pi and centroid at (2, 0)."""
    s = 2.0 * np.sqrt(np.pi / np.sqrt(3.0))
    h = np.sqrt(3.0) * s / 2.0
    return Polygon(
        [
            [CENTER[0], CENTER[1] + 2.0 * h / 3.0],
            [CENTER[0] - s / 2.0, CENTER[1] - h / 3.0],
            [CENTER[0] + s / 2.0, CENTER[1] - h / 3.0],
        ]
    )


def _opened_triangle_unscaled() -> Polygon:
    """Morphological opening of the triangle by a disk of radius OPENING_RADIUS."""
    triangle = equilateral_triangle()
    return triangle.buffer(-OPENING_RADIUS, quad_segs=BUFFER_RESOLUTION).buffer(
        OPENING_RADIUS, quad_segs=BUFFER_RESOLUTION
    )


def opened_triangle_set() -> Polygon:
    """Opened triangle rescaled to area pi and recentered at (2, 0)."""
    opened = _opened_triangle_unscaled()
    scale = np.sqrt(TARGET_AREA / opened.area)
    opened = affinity.scale(opened, xfact=scale, yfact=scale, origin="centroid")
    shift = CENTER - np.array([opened.centroid.x, opened.centroid.y])
    return affinity.translate(opened, xoff=shift[0], yoff=shift[1])


def opened_triangle_reach() -> float:
    """Positive reach r0 of the opened triangle's complement after rescaling (~0.309)."""
    return OPENING_RADIUS * np.sqrt(TARGET_AREA / _opened_triangle_unscaled().area)


def equal_area_initial_sets() -> list[InitialSet]:
    """Circle, opened triangle, and triangle, in the paper's plotting order."""
    return [
        InitialSet("disk", "Circle", disk_set()),
        InitialSet("opened", "Opened triangle", opened_triangle_set()),
        InitialSet("triangle", "Triangle", equilateral_triangle()),
    ]


def star_set(outer_radius: float = 1.0) -> Polygon:
    """Regular five-pointed star (10 vertices) centered at (2, 0), used in Figure 1."""
    inner_radius = outer_radius * np.sin(np.pi / 10.0) / np.sin(3.0 * np.pi / 10.0)
    angles = np.arange(10) * np.pi / 5.0
    radii = np.where(np.arange(10) % 2 == 0, outer_radius, inner_radius)
    return Polygon(CENTER + np.column_stack((radii * np.cos(angles), radii * np.sin(angles))))


def sample_disk(rng: np.random.Generator, n: int, radius: float = DISK_RADIUS) -> np.ndarray:
    """Uniform samples from the disk of the given radius centered at (2, 0)."""
    r = radius * np.sqrt(rng.random(n))
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    return CENTER + np.column_stack((r * np.cos(angle), r * np.sin(angle)))


def sample_uniform_polygon(
    rng: np.random.Generator,
    geom: Polygon,
    n: int,
    batch_size: int | None = None,
) -> np.ndarray:
    """Uniform samples from a polygon by rejection from its bounding box."""
    minx, miny, maxx, maxy = geom.bounds
    if batch_size is None:
        accept_rate = max(geom.area / ((maxx - minx) * (maxy - miny)), 1e-3)
        batch_size = max(1024, int(np.ceil(1.4 * n / accept_rate)))
    path = MplPath(np.asarray(geom.exterior.coords))
    accepted: list[np.ndarray] = []
    count = 0
    while count < n:
        candidates = rng.uniform([minx, miny], [maxx, maxy], size=(batch_size, 2))
        chosen = candidates[path.contains_points(candidates, radius=1e-12)]
        if chosen.size:
            accepted.append(chosen)
            count += chosen.shape[0]
    return np.vstack(accepted)[:n]


def boundary_points(geom: Polygon, n_points: int) -> np.ndarray:
    """n_points equally spaced (in arc length) points on the polygon boundary."""
    ring = LineString(geom.exterior.coords)
    distances = np.linspace(0.0, ring.length, n_points, endpoint=False)
    return np.array([[p.x, p.y] for p in (ring.interpolate(float(d)) for d in distances)], dtype=float)


def _project_to_boundary(points: np.ndarray, geom: Polygon, chunk_size: int = 512) -> np.ndarray:
    starts = np.asarray(geom.exterior.coords[:-1], dtype=float)
    segs = np.roll(starts, -1, axis=0) - starts
    seg_norm2 = np.sum(segs * segs, axis=1)
    seg_norm2[seg_norm2 == 0.0] = 1.0

    out = np.empty_like(points)
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        diff = chunk[:, None, :] - starts[None, :, :]
        tau = np.clip(np.einsum("nsi,si->ns", diff, segs) / seg_norm2[None, :], 0.0, 1.0)
        candidates = starts[None, :, :] + tau[:, :, None] * segs[None, :, :]
        nearest = np.argmin(np.sum((candidates - chunk[:, None, :]) ** 2, axis=2), axis=1)
        out[start : start + chunk.shape[0]] = candidates[np.arange(chunk.shape[0]), nearest]
    return out


def project_to_set(points: np.ndarray, geom: Polygon) -> np.ndarray:
    """Euclidean projection onto a polygon (points inside are left unchanged)."""
    inside = MplPath(np.asarray(geom.exterior.coords)).contains_points(points, radius=1e-12)
    projected = points.copy()
    if np.any(~inside):
        projected[~inside] = _project_to_boundary(points[~inside], geom)
    return projected
