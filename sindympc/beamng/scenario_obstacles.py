"""Spawn shared scenario obstacles as BeamNG procedural cylinders."""

from __future__ import annotations

from typing import Sequence, Tuple

from beamngpy import ProceduralCylinder, Scenario

from sindympc.config.scenario_obstacles import (
    BEAMNG_OBSTACLE_ANNOTATION,
    BEAMNG_OBSTACLE_HEIGHT_M,
    ObstacleDisk,
)

ObstacleList = Sequence[ObstacleDisk]


def add_obstacle_cylinders(
    scenario: Scenario,
    obstacles: ObstacleList,
    *,
    ground_z: float,
    height_m: float = BEAMNG_OBSTACLE_HEIGHT_M,
    annotation: str = BEAMNG_OBSTACLE_ANNOTATION,
) -> int:
    """Add one procedural cylinder per disk; returns count spawned."""
    count = 0
    gz = float(ground_z)
    h = float(height_m)

    for idx, (ox, oy, radius) in enumerate(obstacles):
        mesh = ProceduralCylinder(
            pos=(float(ox), float(oy), gz + h * 0.5),
            radius=float(radius),
            height=h,
            name=f"sindympc_obstacle_{idx}",
            annotation=annotation,
        )
        scenario.add_procedural_mesh(mesh)
        count += 1
    return count
