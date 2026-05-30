"""Time-varying circular obstacles for closed-loop simulation and replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from sindympc.utils.obstacles import Obstacle

Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class MovingObstacle:
    """Disk with constant velocity in world frame (m/s)."""

    x0: float
    y0: float
    radius: float
    vx: float
    vy: float

    def center_at(self, t: float) -> Vec2:
        return self.x0 + self.vx * t, self.y0 + self.vy * t

    def disk_at(self, t: float) -> Obstacle:
        x, y = self.center_at(t)
        return (x, y, self.radius)


@dataclass
class DynamicObstacleField:
    """Collection of movers; query positions at simulation time."""

    movers: List[MovingObstacle]

    def disks_at_time(self, t: float) -> List[Obstacle]:
        return [m.disk_at(t) for m in self.movers]

    def disks_at_step(self, step: int, dt: float) -> List[Obstacle]:
        return self.disks_at_time(step * dt)

    def combined_with_static(
        self,
        static_obstacles: Sequence[Obstacle],
        step: int,
        dt: float,
    ) -> List[Obstacle]:
        return list(static_obstacles) + self.disks_at_step(step, dt)
