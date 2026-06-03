"""Geometry helpers for the quadratic-dynamics experiments."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon


R = 1.0
TARGET_AREA = np.pi
CENTER = np.array([2.0, 0.0])
R_OPEN = 0.3
DISK_RESOLUTION = 512
BUFFER_RESOLUTION = 96


@dataclass(frozen=True)
class InitialSet:
    name: str
    label: str
    geom: Polygon
    color: str


def disk_set() -> Polygon:
    return Point(CENTER).buffer(R, resolution=DISK_RESOLUTION)


def equilateral_triangle() -> Polygon:
    """Equilateral triangle with area pi and centroid exactly at (2,0)."""
    s = 2.0 * np.sqrt(np.pi / np.sqrt(3.0))
    h = np.sqrt(3.0) * s / 2.0
    vertices = np.array(
        [
            [CENTER[0], CENTER[1] + 2.0 * h / 3.0],
            [CENTER[0] - s / 2.0, CENTER[1] - h / 3.0],
            [CENTER[0] + s / 2.0, CENTER[1] - h / 3.0],
        ]
    )
    return Polygon(vertices)


def recenter_to(geom: Polygon, desired_center: np.ndarray = CENTER) -> Polygon:
    c = np.array([geom.centroid.x, geom.centroid.y])
    shift = desired_center - c
    return affinity.translate(geom, xoff=shift[0], yoff=shift[1])


def rescale_area_about_centroid(geom: Polygon, desired_area: float = TARGET_AREA) -> Polygon:
    scale = np.sqrt(desired_area / geom.area)
    return affinity.scale(geom, xfact=scale, yfact=scale, origin="centroid")


def opened_triangle_set() -> Polygon:
    tri = equilateral_triangle()
    opened = tri.buffer(-R_OPEN, resolution=BUFFER_RESOLUTION).buffer(R_OPEN, resolution=BUFFER_RESOLUTION)
    opened = rescale_area_about_centroid(opened, TARGET_AREA)
    return recenter_to(opened, CENTER)


def min_x(geom: Polygon) -> float:
    return float(min(p[0] for p in geom.exterior.coords))


def max_x(geom: Polygon) -> float:
    return float(max(p[0] for p in geom.exterior.coords))


def check_geometry(initial_sets: list[InitialSet]) -> None:
    for item in initial_sets:
        if item.geom.area <= 0.0:
            raise ValueError(f"{item.name} has non-positive area.")
        if min_x(item.geom) <= 0.0:
            raise ValueError(f"{item.name} is not entirely in x > 0.")


def build_equal_area_initial_sets() -> list[InitialSet]:
    sets = [
        InitialSet("disk", "Disk", disk_set(), "#c1121f"),
        InitialSet("triangle", "Triangle", equilateral_triangle(), "#1d3557"),
        InitialSet("opened", "Opened triangle", opened_triangle_set(), "#2a9d8f"),
    ]
    check_geometry(sets)
    return sets


def boundary_points(geom: Polygon, n_points: int) -> np.ndarray:
    ring = LineString(geom.exterior.coords)
    distances = np.linspace(0.0, ring.length, n_points, endpoint=False)
    points = [ring.interpolate(float(distance)) for distance in distances]
    return np.array([[point.x, point.y] for point in points], dtype=float)


def polygon_patch(geom: Polygon, **kwargs) -> MplPolygon:
    coords = np.asarray(geom.exterior.coords)
    return MplPolygon(coords, closed=True, **kwargs)
