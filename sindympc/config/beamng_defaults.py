"""BeamNG world-frame defaults aligned with ``data/trajectory.jsonl`` telemetry.

Coordinates are absolute simulator (x, y) in meters, matching SINDy training data.
Import these in planners, MPC scripts, and dynamics clamps for final BeamNG tests.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# First / last trajectory samples (tick 5 and 15000 in trajectory.jsonl)
BEAMNG_START_XY: Tuple[float, float] = (181.509, 4.607)
BEAMNG_GOAL_XY: Tuple[float, float] = (291.360, -46.020)

# Nominal sample period in collected logs (100 ms between trajectory rows)
BEAMNG_DT: float = 0.1

# Workspace padding for RRT (x, y) — covers the driven corridor with margin
BEAMNG_X_BOUNDS: Tuple[float, float] = (165.0, 310.0)
BEAMNG_Y_BOUNDS: Tuple[float, float] = (-60.0, 20.0)

# SINDy runtime clamps (MPC/simulation safety): keep the plant inside the
# corridor you care about, otherwise the learned model can drift into regions
# with no data support and destabilize MPC.
BEAMNG_STATE_LOW = np.array([150.0, -60.0, -np.pi, -2.0])
BEAMNG_STATE_HIGH = np.array([305.0, 20.0, np.pi, 30.0])
BEAMNG_CONTROL_LOW = np.array([0.0, -1.0])   # throttle, steering in log
BEAMNG_CONTROL_HIGH = np.array([1.0, 1.0])

# RRT / MPC tuning at highway scale (~110 m leg)
BEAMNG_RRT_STEP_SIZE: float = 4.0
BEAMNG_GOAL_THRESHOLD_M: float = 5.0
BEAMNG_GOAL_SLOW_RADIUS_M: float = 20.0
BEAMNG_TARGET_SPEED_MPS: float = 5.0
BEAMNG_INITIAL_SPEED_MPS: float = 0.5
