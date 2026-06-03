"""Polyline helpers for RRT / MPC reference paths."""

from __future__ import annotations

import math
from typing import List, Tuple

Waypoint = Tuple[float, float]


def chaikin_smooth(path: List[Waypoint], iterations: int = 2) -> List[Waypoint]:
    """Chaikin corner-cutting smooth; keeps endpoints fixed."""
    if len(path) < 3 or iterations <= 0:
        return list(path)
    out = list(path)
    for _ in range(iterations):
        smoothed: List[Waypoint] = [out[0]]
        for i in range(len(out) - 1):
            x0, y0 = out[i]
            x1, y1 = out[i + 1]
            smoothed.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            smoothed.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        smoothed.append(out[-1])
        out = smoothed
    return out


def smooth_path_moving_average(
    path: List[Waypoint],
    passes: int = 2,
) -> List[Waypoint]:
    """Light moving-average smooth; keeps endpoints fixed."""
    if len(path) < 3:
        return list(path)
    out = list(path)
    for _ in range(passes):
        smoothed = [out[0]]
        for i in range(1, len(out) - 1):
            x = 0.25 * out[i - 1][0] + 0.5 * out[i][0] + 0.25 * out[i + 1][0]
            y = 0.25 * out[i - 1][1] + 0.5 * out[i][1] + 0.25 * out[i + 1][1]
            smoothed.append((x, y))
        smoothed.append(out[-1])
        out = smoothed
    return out


def densify_polyline(path: List[Waypoint], spacing_m: float = 3.0) -> List[Waypoint]:
    """Resample a polyline at roughly uniform arc-length spacing [m]."""
    if len(path) < 2 or spacing_m <= 0:
        return list(path)

    pts = list(path)
    seg_lens: list[float] = []
    for i in range(len(pts) - 1):
        seg_lens.append(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
    total = sum(seg_lens)
    if total < spacing_m:
        return pts

    n_new = max(2, int(math.ceil(total / spacing_m)) + 1)
    targets = [i * total / (n_new - 1) for i in range(n_new)]

    dense: List[Waypoint] = [pts[0]]
    seg_idx = 0
    acc = 0.0
    for s in targets[1:-1]:
        while seg_idx < len(seg_lens) and acc + seg_lens[seg_idx] < s:
            acc += seg_lens[seg_idx]
            seg_idx += 1
        if seg_idx >= len(seg_lens):
            break
        local = (s - acc) / seg_lens[seg_idx] if seg_lens[seg_idx] > 1e-9 else 0.0
        x = pts[seg_idx][0] + local * (pts[seg_idx + 1][0] - pts[seg_idx][0])
        y = pts[seg_idx][1] + local * (pts[seg_idx + 1][1] - pts[seg_idx][1])
        dense.append((x, y))
    dense.append(pts[-1])
    return dense
