"""Shared utilities for reachable-set experiments."""

from .quadratic_flow import flow, flow_jacobian
from .quaddyn_geometry import (
    InitialSet,
    build_equal_area_initial_sets,
    boundary_points,
    max_x,
    min_x,
    polygon_patch,
)
from .quaddyn_sampling import (
    endpoint_center_and_Q,
    project_points_to_set,
    run_adversarial_sampling,
    run_uniform_sampling,
    sample_uniform_polygon,
    stable_seed_offset,
)
from .support_estimators import (
    approximate_boundary_hausdorff,
    approximate_hausdorff,
    christoffel_support_hausdorff,
    convex_hull_polygon,
)

__all__ = [
    "InitialSet",
    "approximate_boundary_hausdorff",
    "approximate_hausdorff",
    "boundary_points",
    "build_equal_area_initial_sets",
    "christoffel_support_hausdorff",
    "convex_hull_polygon",
    "endpoint_center_and_Q",
    "flow",
    "flow_jacobian",
    "max_x",
    "min_x",
    "polygon_patch",
    "project_points_to_set",
    "run_adversarial_sampling",
    "run_uniform_sampling",
    "sample_uniform_polygon",
    "stable_seed_offset",
]
