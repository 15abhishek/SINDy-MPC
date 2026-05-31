"""Random circular obstacles for RRT / MPC scenarios."""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

Obstacle = Tuple[float, float, float]  # (x, y, radius) [m]

DEFAULT_NUM_OBSTACLES = 10
DEFAULT_RADIUS_RANGE = (10.0, 30.0)
DEFAULT_MIN_DIST_FROM_POINTS = 5.0
DEFAULT_MAX_TRIES = 5000
DEFAULT_SEED = random.randint(0, 1000000)


def euclidean_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Euclidean distance between two (x, y) points [m]."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def sample_random_obstacles(
    num: int,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    radius_range: Tuple[float, float],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    *,
    min_dist_from_points: float = DEFAULT_MIN_DIST_FROM_POINTS,
    seed: int = DEFAULT_SEED,
    max_tries: int = DEFAULT_MAX_TRIES,
) -> List[Obstacle]:
    """Sample circular obstacles inside bounds, away from start and goal.

    Each obstacle is ``(x, y, radius)``. Centers are rejected if the circle
    edge lies within ``min_dist_from_points`` of ``start`` or ``goal``.

    Args:
        num: Target number of obstacles.
        x_bounds: ``(x_min, x_max)`` in meters.
        y_bounds: ``(y_min, y_max)`` in meters.
        radius_range: ``(r_min, r_max)`` in meters.
        start: Start ``(x, y)`` to keep clear.
        goal: Goal ``(x, y)`` to keep clear.
        min_dist_from_points: Minimum center-to-point distance minus radius.
        seed: RNG seed for reproducibility.
        max_tries: Rejection-sampling attempt limit.

    Returns:
        List of up to ``num`` obstacles (may be shorter if placement fails).
    """
    rng = random.Random(seed)
    obs: List[Obstacle] = []
    tries = 0
    while len(obs) < num and tries < max_tries:
        x = rng.uniform(x_bounds[0], x_bounds[1])
        y = rng.uniform(y_bounds[0], y_bounds[1])
        r = rng.uniform(radius_range[0], radius_range[1])
        center = (x, y)
        if (
            euclidean_distance(center, start) > min_dist_from_points + r
            and euclidean_distance(center, goal) > min_dist_from_points + r
        ):
            obs.append((x, y, r))
        tries += 1
    if len(obs) < num:
        print(f"Warning: Only placed {len(obs)} obstacles (couldn't fit {num})")
    return obs


def nearest_obstacles(
    position: Tuple[float, float],
    obstacles: Sequence[Obstacle],
    k: int,
) -> List[Obstacle]:
    """Return up to ``k`` obstacles with smallest center-to-point distance."""
    if k <= 0 or not obstacles:
        return []
    px, py = position

    def _dist(obs: Obstacle) -> float:
        ox, oy, _ = obs
        return math.hypot(ox - px, oy - py)

    ranked = sorted(obstacles, key=_dist)
    return list(ranked[: min(k, len(ranked))])


def pad_obstacle_list(
    obstacles: Sequence[Obstacle],
    count: int,
    *,
    pad_center: Tuple[float, float] = (1.0e6, 1.0e6),
    pad_radius: float = 0.05,
) -> List[Obstacle]:
    """Pad or truncate to fixed length (MPC QP size is fixed at init)."""
    out = list(obstacles[:count])
    while len(out) < count:
        out.append((pad_center[0], pad_center[1], pad_radius))
    return out
