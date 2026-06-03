"""Reference trajectory along a polyline (shared by SQP-MPC and MPPI)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

Waypoint = Tuple[float, float]


def cross_track_distance(pos: np.ndarray, waypoints: np.ndarray) -> float:
    """Minimum distance from ``pos`` (2,) to the path polyline [m]."""
    best = np.inf
    for i in range(waypoints.shape[0] - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-12:
            dist = float(np.linalg.norm(pos - p0))
        else:
            t = float(np.clip(np.dot(pos - p0, seg) / seg_len_sq, 0.0, 1.0))
            proj = p0 + t * seg
            dist = float(np.linalg.norm(pos - proj))
        best = min(best, dist)
    return best


def path_length(waypoints: List[Waypoint]) -> float:
    """Total arc length of a waypoint polyline [m]."""
    if len(waypoints) < 2:
        return 0.0
    arr = np.array(waypoints, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def path_tracking_stats(
    executed_xy: np.ndarray,
    waypoints: List[Waypoint],
    final_arc_s: Optional[float],
) -> dict[str, float]:
    """Summary metrics for how closely a trajectory followed a reference path."""
    wp = np.array(waypoints, dtype=np.float64)
    seg_lens = np.linalg.norm(np.diff(wp, axis=0), axis=1) if len(wp) > 1 else []
    total_len = float(np.sum(seg_lens)) if len(seg_lens) else 0.0
    cross = [
        cross_track_distance(executed_xy[i], wp) for i in range(executed_xy.shape[0])
    ]
    progress = (
        100.0 * final_arc_s / total_len
        if final_arc_s is not None and total_len > 1e-6
        else float("nan")
    )
    return {
        "path_length_m": total_len,
        "max_cross_track_m": float(max(cross)) if cross else 0.0,
        "mean_cross_track_m": float(np.mean(cross)) if cross else 0.0,
        "path_progress_pct": progress,
    }


def project_onto_path(
    pos: np.ndarray,
    waypoints: np.ndarray,
    arc_length: np.ndarray,
) -> float:
    """Arc-length coordinate of the closest point on the path polyline."""
    best_s = 0.0
    best_dist_sq = np.inf

    for i in range(waypoints.shape[0] - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))

        if seg_len_sq < 1e-12:
            t = 0.0
            proj = p0
        else:
            t = float(np.clip(np.dot(pos - p0, seg) / seg_len_sq, 0.0, 1.0))
            proj = p0 + t * seg

        dist_sq = float(np.sum((pos - proj) ** 2))
        s = arc_length[i] + t * (arc_length[i + 1] - arc_length[i])

        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_s = s

    return best_s


def heading_along_path(
    s: float,
    arc_length: np.ndarray,
    waypoints: np.ndarray,
    eps: float = 0.2,
) -> float:
    """Path tangent heading at arc-length ``s``."""
    s_lo = max(float(arc_length[0]), s - eps)
    s_hi = min(float(arc_length[-1]), s + eps)
    if s_hi - s_lo < 1e-6:
        s_hi = min(float(arc_length[-1]), s_lo + eps)

    x_lo = np.interp(s_lo, arc_length, waypoints[:, 0])
    y_lo = np.interp(s_lo, arc_length, waypoints[:, 1])
    x_hi = np.interp(s_hi, arc_length, waypoints[:, 0])
    y_hi = np.interp(s_hi, arc_length, waypoints[:, 1])
    return float(np.arctan2(y_hi - y_lo, x_hi - x_lo))


def waypoints_to_reference(
    current_state: jnp.ndarray,
    rrt_waypoints: List[Waypoint],
    *,
    horizon: int,
    dt: float,
    target_speed: float,
    goal_position: Optional[Tuple[float, float]],
    last_arc_s: Optional[float],
    goal_slow_radius: float = 8.0,
) -> Tuple[jnp.ndarray, float]:
    """Build (H+1, 4) reference [x, y, theta, v] and updated arc-length progress."""
    if not rrt_waypoints:
        ref = jnp.tile(current_state[None, :], (horizon + 1, 1))
        return ref, 0.0

    waypoints = np.array(rrt_waypoints, dtype=np.float64)
    pos = np.array(
        [float(current_state[0]), float(current_state[1])], dtype=np.float64
    )

    seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc_length = np.zeros(waypoints.shape[0], dtype=np.float64)
    arc_length[1:] = np.cumsum(seg_lens)

    s_closest = project_onto_path(pos, waypoints, arc_length)
    s_base = s_closest if last_arc_s is None else max(s_closest, last_arc_s)

    lookahead = np.arange(horizon + 1, dtype=np.float64) * dt * target_speed
    s_targets = np.clip(s_base + lookahead, arc_length[0], arc_length[-1])

    x_ref = np.interp(s_targets, arc_length, waypoints[:, 0])
    y_ref = np.interp(s_targets, arc_length, waypoints[:, 1])
    theta_ref = np.array(
        [heading_along_path(float(s), arc_length, waypoints) for s in s_targets],
        dtype=np.float64,
    )

    v_now = float(current_state[3])
    dist_to_goal = (
        float(np.hypot(pos[0] - goal_position[0], pos[1] - goal_position[1]))
        if goal_position is not None
        else np.inf
    )

    v_ref = np.empty(horizon + 1, dtype=np.float64)
    cruise = dist_to_goal > 25.0
    for k in range(horizon + 1):
        if cruise:
            v_ref[k] = target_speed
        else:
            alpha = min(1.0, (k + 1) / max(horizon // 2, 1))
            v_ref[k] = max(v_now, (1.0 - alpha) * v_now + alpha * target_speed)

    if goal_position is not None:
        s_max = float(arc_length[-1])
        path_len = max(s_max - float(arc_length[0]), 1e-6)
        path_progress = (s_base - float(arc_length[0])) / path_len
        near_path_end = path_progress > 0.85

        gx, gy = goal_position
        goal_theta = heading_along_path(s_max, arc_length, waypoints)

        # Gentle geographic slowdown only inside slow radius (poster: was too cautious at 20 m).
        if dist_to_goal < goal_slow_radius:
            slow = np.clip(dist_to_goal / max(goal_slow_radius, 1e-3), 0.55, 1.0)
            v_ref *= slow

        if near_path_end:
            blend_len = 1.5
            for k in range(horizon + 1):
                blend = np.clip((s_targets[k] - (s_max - blend_len)) / blend_len, 0.0, 1.0)
                x_ref[k] = (1.0 - blend) * x_ref[k] + blend * gx
                y_ref[k] = (1.0 - blend) * y_ref[k] + blend * gy
                theta_ref[k] = (1.0 - blend) * theta_ref[k] + blend * goal_theta
                v_ref[k] = (1.0 - blend) * v_ref[k]

            if dist_to_goal < 5.0:
                slow = np.clip(dist_to_goal / 5.0, 0.25, 1.0)
                v_ref *= slow
            if dist_to_goal < 1.0:
                lock = np.clip((1.0 - dist_to_goal) / 1.0, 0.0, 1.0)
                x_ref = (1.0 - lock) * x_ref + lock * gx
                y_ref = (1.0 - lock) * y_ref + lock * gy
                theta_ref = (1.0 - lock) * theta_ref + lock * goal_theta
                v_ref = (1.0 - lock) * v_ref

    ref = np.stack([x_ref, y_ref, theta_ref, v_ref], axis=1)
    return jnp.array(ref), float(s_base)
