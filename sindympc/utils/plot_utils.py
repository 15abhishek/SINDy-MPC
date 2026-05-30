"""Shared matplotlib axis limits for scenario plots."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from sindympc.config.beamng_defaults import (
    BEAMNG_GOAL_XY,
    BEAMNG_SCENARIO_MARGIN_M,
    BEAMNG_START_XY,
    beamng_scenario_xy_bounds,
)


def scenario_xy_limits(
    *,
    extra_xy: Optional[Sequence[Tuple[float, float]]] = None,
    trajectory_xy: Optional[np.ndarray] = None,
    margin_m: float = BEAMNG_SCENARIO_MARGIN_M,
    pad_m: float = 8.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Equal-aspect limits covering start, goal, and optional path samples."""
    (x_lo, x_hi), (y_lo, y_hi) = beamng_scenario_xy_bounds(margin_m)

    xs = [BEAMNG_START_XY[0], BEAMNG_GOAL_XY[0], x_lo, x_hi]
    ys = [BEAMNG_START_XY[1], BEAMNG_GOAL_XY[1], y_lo, y_hi]

    if extra_xy:
        for x, y in extra_xy:
            xs.append(float(x))
            ys.append(float(y))
    if trajectory_xy is not None and trajectory_xy.size > 0:
        xs.extend(trajectory_xy[:, 0].tolist())
        ys.extend(trajectory_xy[:, 1].tolist())

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    half = 0.5 * max(span_x, span_y, 20.0) + pad_m
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    return ((cx - half, cx + half), (cy - half, cy + half))


def apply_scenario_axes(
    ax,
    *,
    extra_xy: Optional[Sequence[Tuple[float, float]]] = None,
    trajectory_xy: Optional[np.ndarray] = None,
) -> None:
    """Set square-ish equal limits so routes are not stretched into rectangles."""
    (x_lo, x_hi), (y_lo, y_hi) = scenario_xy_limits(
        extra_xy=extra_xy,
        trajectory_xy=trajectory_xy,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
