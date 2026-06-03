"""Shared static obstacles for RRT, JAX sim, matplotlib, and BeamNG (1:1 XY frame).

Obstacles are circular disks ``(x, y, radius_m)`` in the same world coordinates as
``BEAMNG_START_XY`` / ``data/trajectory.jsonl``. Edit ``SCENARIO_OBSTACLES`` to add
manual blockers; RRT collision checks and BeamNG procedural cylinders use this list.
"""

from __future__ import annotations

from typing import List, Tuple

ObstacleDisk = Tuple[float, float, float]

# Default route: start (-122, -188) → goal (-90.3, -92.6) on gridmap_v2 (~z=100 m).
# Radii are modest so the planner detours but the corridor stays drivable.
SCENARIO_OBSTACLES: List[ObstacleDisk] = [
    (-150.0 , -150.0, 7.0),
    (-165.0, -125.0, 5.0),
    (-135.0, -125.0, 7.0),
    (-153.0, -80.0, 4.5),
    (-147.0, -80.0, 4.5),
    (-147.0, -53.0, 1.0),       # Goal Bars
    (-153.0, -53.0, 1.0),       # Goal Bars
    (-147.0, -47.0, 1.0),       # Goal Bars
    (-153.0, -47.0, 1.0),       # Goal Bars
]

# Nominal terrain Z on gridmap_v2 along the start→goal leg (from live telemetry).
BEAMNG_OBSTACLE_GROUND_Z: float = 100.0

# One procedural cylinder per disk; annotation selects the in-engine material.
BEAMNG_OBSTACLE_HEIGHT_M: float = 6.0
BEAMNG_OBSTACLE_ANNOTATION: str = "BUILDINGS"
