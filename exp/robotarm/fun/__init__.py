"""Shared robot-arm dynamics and experiment utilities."""

from typing import Any

__all__ = [
    "DEFAULT_GAMMA",
    "DEFAULT_KD",
    "DEFAULT_KP",
    "DEFAULT_LAMBDA",
    "DEFAULT_SIGMA",
    "DEFAULT_TAU_LIMIT",
    "TRACKING_CONTROLLER_NAME",
    "MuJoCoNLinkArm",
    "make_n_link_arm_xml",
    "reference_parameters",
    "reference_trajectory",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DEFAULT_GAMMA",
        "DEFAULT_KD",
        "DEFAULT_KP",
        "DEFAULT_LAMBDA",
        "DEFAULT_SIGMA",
        "DEFAULT_TAU_LIMIT",
        "TRACKING_CONTROLLER_NAME",
        "MuJoCoNLinkArm",
        "make_n_link_arm_xml",
        "reference_parameters",
        "reference_trajectory",
    }:
        from .mujoco_n_link_arm import (
            DEFAULT_GAMMA,
            DEFAULT_KD,
            DEFAULT_KP,
            DEFAULT_LAMBDA,
            DEFAULT_SIGMA,
            DEFAULT_TAU_LIMIT,
            TRACKING_CONTROLLER_NAME,
            MuJoCoNLinkArm,
            make_n_link_arm_xml,
            reference_parameters,
            reference_trajectory,
        )

        return {
            "DEFAULT_GAMMA": DEFAULT_GAMMA,
            "DEFAULT_KD": DEFAULT_KD,
            "DEFAULT_KP": DEFAULT_KP,
            "DEFAULT_LAMBDA": DEFAULT_LAMBDA,
            "DEFAULT_SIGMA": DEFAULT_SIGMA,
            "DEFAULT_TAU_LIMIT": DEFAULT_TAU_LIMIT,
            "TRACKING_CONTROLLER_NAME": TRACKING_CONTROLLER_NAME,
            "MuJoCoNLinkArm": MuJoCoNLinkArm,
            "make_n_link_arm_xml": make_n_link_arm_xml,
            "reference_parameters": reference_parameters,
            "reference_trajectory": reference_trajectory,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
