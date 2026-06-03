"""Metrics for BeamNG closed-loop and open-loop SINDy benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def compute_max_steps(
    route_length_m: float,
    target_speed_mps: float,
    dt: float,
    *,
    margin: int = 150,
    min_steps: int = 200,
) -> int:
    """Tick budget from route length and target speed (used by ``07`` / ``08``)."""
    speed = max(float(target_speed_mps), 0.5)
    base = int(np.ceil(route_length_m / (speed * dt))) + margin
    # Extra time when target is low or achieved speed may lag setpoint.
    if speed < 4.0:
        base = int(base * (4.0 / speed) ** 0.35)
    return max(min_steps, base)


def cross_track_error(
    trajectory_xy: np.ndarray,
    reference_path: Sequence[tuple[float, float]] | np.ndarray,
) -> np.ndarray:
    """Per-point lateral distance from ``trajectory_xy`` to polyline ``reference_path`` [m]."""
    traj = np.asarray(trajectory_xy, dtype=np.float64).reshape(-1, 2)
    ref = np.asarray(reference_path, dtype=np.float64).reshape(-1, 2)
    if traj.size == 0 or ref.shape[0] < 2:
        return np.zeros(traj.shape[0], dtype=np.float64)

    seg = ref[1:] - ref[:-1]
    seg_len2 = np.sum(seg**2, axis=1)
    errors = np.empty(traj.shape[0], dtype=np.float64)
    for i, p in enumerate(traj):
        best = np.inf
        for j, d in enumerate(seg):
            if seg_len2[j] < 1e-12:
                dist = float(np.linalg.norm(p - ref[j]))
            else:
                t = float(np.clip(np.dot(p - ref[j], d) / seg_len2[j], 0.0, 1.0))
                proj = ref[j] + t * d
                dist = float(np.linalg.norm(p - proj))
            best = min(best, dist)
        errors[i] = best
    return errors


def load_jsonl_log(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_run(
    log_path: Path | str | None,
    reference_path: Sequence[tuple[float, float]] | np.ndarray,
    goal_xy: tuple[float, float],
    *,
    target_speed_mps: float,
    goal_threshold_m: float = 5.0,
    goal_slow_radius_m: float = 20.0,
    slow_speed_mps: float = 0.15,
    dt: float = 0.1,
    solve_times: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Aggregate metrics from a BeamNG JSONL log (or empty dict if missing)."""
    goal = np.asarray(goal_xy, dtype=np.float64)
    out: dict[str, Any] = {
        "target_speed_mps": float(target_speed_mps),
        "mean_speed_mps": 0.0,
        "final_goal_error_m": float("inf"),
        "success": False,
        "ticks": 0,
        "duration_s": 0.0,
        "mean_cross_track_error_m": float("nan"),
        "max_cross_track_error_m": float("nan"),
        "mean_solve_time_s": float("nan"),
        "mean_throttle": float("nan"),
        "mean_brake": float("nan"),
        "mean_mpc_accel": float("nan"),
    }
    if log_path is None or not Path(log_path).is_file():
        return out

    records = load_jsonl_log(log_path)
    if not records:
        return out

    states = np.array([r["state"] for r in records], dtype=np.float64)
    xy = states[:, :2]
    speeds = states[:, 3]
    throttles = np.array([r.get("throttle", 0.0) for r in records], dtype=np.float64)
    brakes = np.array([r.get("brake", 0.0) for r in records], dtype=np.float64)
    mpc_a = np.array(
        [r.get("control_mpc", [0.0, 0.0])[0] for r in records], dtype=np.float64
    )

    ct = cross_track_error(xy, reference_path)
    final = states[-1]
    final_err = float(np.linalg.norm(final[:2] - goal))
    success = final_err < goal_threshold_m
    if not success and final_err < goal_slow_radius_m and final[3] < slow_speed_mps:
        success = True

    out.update(
        {
            "mean_speed_mps": float(np.mean(speeds)),
            "final_goal_error_m": final_err,
            "success": success,
            "ticks": len(records),
            "duration_s": len(records) * dt,
            "mean_cross_track_error_m": float(np.mean(ct)),
            "max_cross_track_error_m": float(np.max(ct)),
            "mean_throttle": float(np.mean(throttles)),
            "mean_brake": float(np.mean(brakes)),
            "mean_mpc_accel": float(np.mean(mpc_a)),
        }
    )
    if solve_times is not None and len(solve_times) > 0:
        out["mean_solve_time_s"] = float(np.mean(solve_times))
    return out


def write_summary_csv(rows: Iterable[Mapping[str, Any]], path: Path | str) -> Path:
    """Write benchmark rows to CSV; returns resolved path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        resolved.write_text("", encoding="utf-8")
        return resolved
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with resolved.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def format_summary_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Poster-friendly markdown table from summary rows."""
    if not rows:
        return "_No benchmark runs._"

    if rows[0].get("mode") == "offline_openloop":
        cols = [
            "speed_bin",
            "n_samples",
            "bicycle_rmse_xy",
            "bicycle_rmse_v",
            "sindy_rmse_xy",
            "sindy_rmse_v",
        ]
    else:
        cols = [
            "controller",
            "dynamics",
            "target_speed_mps",
            "mean_speed_mps",
            "final_goal_error_m",
            "success",
            "mean_cross_track_error_m",
        ]

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            val = r.get(c, "")
            if c == "success":
                cells.append("yes" if val else "no")
            elif isinstance(val, float):
                cells.append(f"{val:.4g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
