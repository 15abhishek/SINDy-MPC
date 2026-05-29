"""Load BeamNG trajectory and control logs with timestamp alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

N_STATES = 4
N_CONTROLS = 2

HeadingConvention = Literal["velocity", "quaternion"]

# Use velocity heading when speed exceeds this [m/s] (kinematic bicycle convention).
HEADING_VELOCITY_MIN_SPEED = 0.3


def compute_median_dt(timestamps_ms: np.ndarray) -> float:
    """Median sample period [s] from monotonic ``timestamp_ms``."""
    ts = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    if len(ts) < 2:
        raise ValueError("Need at least two timestamps to compute dt.")
    diffs = np.diff(ts)
    if np.any(diffs <= 0):
        raise ValueError("timestamps_ms must be strictly increasing.")
    return float(np.median(diffs) / 1000.0)


def load_controls_by_timestamp(control_path: Path) -> dict[int, list[float]]:
    """Index control records by ``timestamp_ms`` (controls log at every tick)."""
    controls_by_ts: dict[int, list[float]] = {}
    with control_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ts = int(data["timestamp_ms"])
            controls_by_ts[ts] = [
                float(data.get("throttle", 0.0)),
                float(data.get("steering", 0.0)),
            ]
    return controls_by_ts


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    return float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy**2 + qz**2),
        )
    )


def apply_heading_convention(
    states: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    convention: HeadingConvention = "velocity",
    min_speed: float = HEADING_VELOCITY_MIN_SPEED,
) -> np.ndarray:
    """Align ``theta`` with the kinematic bicycle heading convention.

    The JAX bicycle model uses ``ẋ = v cos θ``, ``ẏ = v sin θ`` with θ equal to
    the **velocity heading** ``atan2(vy, vx)``. BeamNG quaternion yaw can differ
    by a fixed offset (often ±π/2). This replaces θ with the velocity heading
    wherever ``v >= min_speed``.
    """
    states = np.asarray(states, dtype=np.float64).copy()
    v = np.hypot(vx, vy)
    theta_q = states[:, 2]
    theta_v = np.arctan2(vy, vx)
    moving = v >= min_speed
    states[moving, 2] = theta_v[moving]
    return states


def load_aligned_telemetry(
    trajectory_path: Path,
    control_path: Path,
    *,
    heading: HeadingConvention = "velocity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load aligned ``states``, ``controls``, ``timestamps_ms``, and median ``dt``.

    Trajectory is sampled every 5 ticks; controls every 1 tick. Rows are kept
    only when ``timestamp_ms`` matches between files.

    Args:
        heading: ``"velocity"`` (default, bicycle-consistent) or ``"quaternion"``.
    """
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")
    if not control_path.is_file():
        raise FileNotFoundError(f"Controls not found: {control_path}")

    controls_by_ts = load_controls_by_timestamp(control_path)

    states_list: list[list[float]] = []
    controls_list: list[list[float]] = []
    timestamps_ms: list[int] = []
    vx_list: list[float] = []
    vy_list: list[float] = []
    missing_ts: list[int] = []

    with trajectory_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ts = int(data["timestamp_ms"])

            control = controls_by_ts.get(ts)
            if control is None:
                missing_ts.append(ts)
                continue

            timestamps_ms.append(ts)

            x, y, _z = data["pos"]
            vx, vy, _vz = data["velocity"]
            v = float(np.hypot(vx, vy))

            qx, qy, qz, qw = data["rot_quat"]
            theta = _yaw_from_quaternion(qx, qy, qz, qw)

            states_list.append([x, y, theta, v])
            controls_list.append(control)
            vx_list.append(float(vx))
            vy_list.append(float(vy))

    if missing_ts:
        raise ValueError(
            f"No control record for {len(missing_ts)} trajectory timestamp(s). "
            f"First missing timestamp_ms: {missing_ts[0]}"
        )
    if not states_list:
        raise ValueError("No aligned trajectory/control samples were loaded.")

    states = np.asarray(states_list, dtype=np.float64)
    controls = np.asarray(controls_list, dtype=np.float64)
    timestamps_ms_arr = np.asarray(timestamps_ms, dtype=np.int64)
    vx_arr = np.asarray(vx_list, dtype=np.float64)
    vy_arr = np.asarray(vy_list, dtype=np.float64)

    if heading == "velocity":
        states = apply_heading_convention(states, vx_arr, vy_arr)
    elif heading != "quaternion":
        raise ValueError(f"Unknown heading convention: {heading!r}")

    dt = compute_median_dt(timestamps_ms_arr)

    if states.shape[1] != N_STATES:
        raise ValueError(f"Expected states with {N_STATES} columns, got {states.shape}")
    if controls.shape[1] != N_CONTROLS:
        raise ValueError(
            f"Expected controls with {N_CONTROLS} columns, got {controls.shape}"
        )
    if len(states) != len(controls):
        raise ValueError(
            f"Aligned row count mismatch: {len(states)} states vs {len(controls)} controls"
        )

    return states, controls, timestamps_ms_arr, dt


def estimate_acceleration(
    states: np.ndarray,
    timestamps_ms: np.ndarray,
) -> np.ndarray:
    """Estimate longitudinal acceleration [m/s²] from speed samples.

    Uses ``np.gradient`` so spacing matches irregular subsets after downsampling.
    """
    v = np.asarray(states[:, 3], dtype=np.float64)
    t = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1) / 1000.0
    if len(v) != len(t):
        raise ValueError("states and timestamps_ms must have equal length.")
    return np.gradient(v, t).astype(np.float64)
