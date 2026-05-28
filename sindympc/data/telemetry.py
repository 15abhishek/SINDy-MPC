"""Load BeamNG trajectory and control logs with timestamp alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sindympc.system_id.sindy_training import N_CONTROLS, N_STATES


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


def load_aligned_telemetry(
    trajectory_path: Path,
    control_path: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load aligned ``states`` and training ``controls`` (throttle, steering).

    Trajectory is sampled every 5 ticks; controls every 1 tick. Rows are kept
    only when ``timestamp_ms`` matches between files.
    """
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")
    if not control_path.is_file():
        raise FileNotFoundError(f"Controls not found: {control_path}")

    controls_by_ts = load_controls_by_timestamp(control_path)

    states_list: list[list[float]] = []
    controls_list: list[list[float]] = []
    timestamps_ms: list[int] = []
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
            theta = float(
                np.arctan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy**2 + qz**2),
                )
            )

            states_list.append([x, y, theta, v])
            controls_list.append(control)

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
    dt = float(np.median(np.diff(timestamps_ms_arr)) / 1000.0)

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

    return states, controls, dt
