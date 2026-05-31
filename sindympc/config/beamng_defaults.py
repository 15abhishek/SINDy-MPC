"""BeamNG world-frame defaults aligned with ``data/trajectory.jsonl`` telemetry.

Coordinates are absolute simulator (x, y) in meters, matching SINDy training data.
Import these in planners, MPC scripts, and dynamics clamps for final BeamNG tests.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# 3D spawn / goal poses (BeamNG world frame, e.g. GridMap v2)
BEAMNG_SPAWN_POS: Tuple[float, float, float] = (-122.011, -187.990, 100.330)
BEAMNG_SPAWN_ROT: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
BEAMNG_GOAL_POS: Tuple[float, float, float] = (-90.316, -92.625, 100.317)
BEAMNG_GOAL_ROT: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0)

# 2D planning / MPC frame (XY slices — controllers and RRT use these only)
BEAMNG_START_XY: Tuple[float, float] = (BEAMNG_SPAWN_POS[0], BEAMNG_SPAWN_POS[1])
BEAMNG_GOAL_XY: Tuple[float, float] = (BEAMNG_GOAL_POS[0], BEAMNG_GOAL_POS[1])

# Default initial yaw: aligned with the start→goal leg (override with BEAMNG_SPAWN_ROT if needed)
BEAMNG_PATH_HEADING: float = math.atan2(
    BEAMNG_GOAL_XY[1] - BEAMNG_START_XY[1],
    BEAMNG_GOAL_XY[0] - BEAMNG_START_XY[0],
)

# Backward-compatible aliases (deprecated spelling)
BeamNG_SPAWN_POS = BEAMNG_SPAWN_POS
BeamNG_SPAWN_ROT = BEAMNG_SPAWN_ROT

# Nominal sample period in collected logs (100 ms between trajectory rows)
BEAMNG_DT: float = 0.1

# Full simulator workspace for RRT sampling (large map)
BEAMNG_X_BOUNDS: Tuple[float, float] = (-500.0, 500.0)
BEAMNG_Y_BOUNDS: Tuple[float, float] = (-500.0, 500.0)

# Margin around start/goal for SINDy clamps, plots, and local obstacle placement
BEAMNG_SCENARIO_MARGIN_M: float = 45.0


def beamng_scenario_xy_bounds(
    margin_m: float = BEAMNG_SCENARIO_MARGIN_M,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Tight (x, y) limits covering the start–goal leg plus ``margin_m``."""
    xs = (BEAMNG_START_XY[0], BEAMNG_GOAL_XY[0])
    ys = (BEAMNG_START_XY[1], BEAMNG_GOAL_XY[1])
    return (
        (min(xs) - margin_m, max(xs) + margin_m),
        (min(ys) - margin_m, max(ys) + margin_m),
    )


def beamng_state_limits(
    margin_m: float = BEAMNG_SCENARIO_MARGIN_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """SINDy / plant state clamps aligned with the current scenario (not the full map)."""
    (x_lo, x_hi), (y_lo, y_hi) = beamng_scenario_xy_bounds(margin_m)
    low = np.array([x_lo, y_lo, -np.pi, -2.0], dtype=np.float64)
    high = np.array([x_hi, y_hi, np.pi, 30.0], dtype=np.float64)
    return low, high


BEAMNG_SCENARIO_X_BOUNDS, BEAMNG_SCENARIO_Y_BOUNDS = beamng_scenario_xy_bounds()
BEAMNG_STATE_LOW, BEAMNG_STATE_HIGH = beamng_state_limits()

BEAMNG_CONTROL_LOW = np.array([0.0, -1.0])   # throttle, steering in log
BEAMNG_CONTROL_HIGH = np.array([1.0, 1.0])

# RRT / MPC tuning
BEAMNG_RRT_STEP_SIZE: float = 0.5
BEAMNG_GOAL_THRESHOLD_M: float = 1.0
BEAMNG_GOAL_SLOW_RADIUS_M: float = 20.0
BEAMNG_TARGET_SPEED_MPS: float = 5.0
BEAMNG_INITIAL_SPEED_MPS: float = 0.1
