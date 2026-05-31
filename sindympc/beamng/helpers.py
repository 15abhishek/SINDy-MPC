"""BeamNG telemetry polling and MPC→actuator mapping for live closed-loop runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from sindympc.data.telemetry import HEADING_VELOCITY_MIN_SPEED

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def heading_from_quat(quat: Sequence[float]) -> float:
    """World-frame heading [rad] from body→world quaternion (HybridVLA convention)."""
    fwd_world = Rotation.from_quat(quat).apply([0.0, -1.0, 0.0])
    return float(math.atan2(fwd_world[1], fwd_world[0]))


def quat_from_heading(heading: float) -> Quat:
    """Quaternion (x, y, z, w) so local -Y aligns with ``heading`` in the XY plane."""
    fwd = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=np.float64)
    rot, _ = Rotation.align_vectors([fwd], [[0.0, -1.0, 0.0]])
    q = rot.as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def poll_vehicle_state_4d(vehicle) -> Tuple[np.ndarray, Vec3, Vec3, int]:
    """Return ``[x, y, theta, v]``, raw velocity, position, and sim time [ms]."""
    vehicle.sensors.poll("state")
    st = vehicle.state
    pos = st["pos"]
    rot = st["rotation"]
    vel = st["vel"]
    sim_time_ms = int(round(float(st["time"]) * 1000.0))

    x, y = float(pos[0]), float(pos[1])
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    v = float(math.sqrt(vx * vx + vy * vy + vz * vz))

    theta_q = heading_from_quat(rot)
    if v >= HEADING_VELOCITY_MIN_SPEED:
        theta = float(math.atan2(vy, vx))
    else:
        theta = theta_q

    state = np.array([x, y, theta, v], dtype=np.float64)
    return state, (vx, vy, vz), (float(pos[0]), float(pos[1]), float(pos[2])), sim_time_ms


@dataclass
class SpeedPID:
    """Convert desired speed [m/s] to throttle/brake (HybridVLA runner pattern)."""

    dt: float
    kp: float = 0.5
    ki: float = 0.3
    kd: float = 0.05
    windup_limit: float = 1.0
    max_delta: float = 0.15

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0

    def __call__(
        self,
        desired_speed: float,
        vel: Sequence[float],
        emergency_brake: float = 0.0,
    ) -> Tuple[float, float]:
        if emergency_brake > 0.3:
            self.reset()
            return 0.0, min(0.6, emergency_brake)

        cur_speed = float(
            math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        )

        if desired_speed < 0.1:
            self.reset()
            return 0.0, float(np.clip(cur_speed * 0.3, 0.0, 0.5))

        err = desired_speed - cur_speed
        self._integral += err * self.dt
        self._integral = float(
            np.clip(self._integral, -self.windup_limit, self.windup_limit)
        )
        derivative = (err - self._prev_err) / self.dt if self.dt > 0 else 0.0
        self._prev_err = err

        output = self.kp * err + self.ki * self._integral + self.kd * derivative
        if output >= 0:
            raw_throttle = float(np.clip(output, 0.0, 0.7))
            raw_brake = 0.0
        else:
            raw_throttle = 0.0
            raw_brake = float(np.clip(-output, 0.0, 0.7))

        throttle = float(
            np.clip(
                raw_throttle,
                self._prev_throttle - self.max_delta,
                self._prev_throttle + self.max_delta,
            )
        )
        brake = float(
            np.clip(
                raw_brake,
                self._prev_brake - self.max_delta,
                self._prev_brake + self.max_delta,
            )
        )
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        self._prev_throttle = throttle
        self._prev_brake = brake
        return throttle, brake


def mpc_control_to_beamng(
    control: Sequence[float],
    state: Sequence[float],
    vel: Sequence[float],
    *,
    dt: float,
    max_acceleration: float,
    max_steering_angle: float,
    speed_pid: SpeedPID,
    max_speed_mps: float = 30.0,
    speed_horizon_s: float = 3.0,
    min_creep_speed_mps: float = 2.0,
) -> Tuple[float, float, float]:
    """Map MPC/MPPI ``[a, delta]`` to BeamNG ``throttle, brake, steering``.

    MPC acceleration is integrated over ``speed_horizon_s`` (not one control ``dt``)
    to form a speed setpoint for the PID. When the vehicle is nearly stopped but
    MPC commands forward acceleration, a minimum creep speed avoids the PID
    deadband that keeps throttle at zero.
    """
    a = float(control[0])
    delta = float(control[1])
    v = float(state[3])

    desired_speed = float(
        np.clip(v + a * speed_horizon_s, 0.0, max_speed_mps)
    )
    if a > 0.05 and v < 1.0:
        desired_speed = max(desired_speed, min_creep_speed_mps)

    throttle, brake = speed_pid(desired_speed, vel)

    # Arcade fallback when PID output is too weak for static friction
    if a > 0.05 and throttle < 0.15:
        throttle = float(
            np.clip(0.25 + 0.45 * (a / max_acceleration), 0.15, 0.75)
        )
        brake = 0.0
    elif a < -0.05:
        throttle = 0.0
        brake = max(
            brake,
            float(np.clip(-a / max_acceleration * 0.55, 0.05, 0.75)),
        )

    steer_norm = float(np.clip(delta / max_steering_angle, -1.0, 1.0))
    steering = -steer_norm
    return throttle, brake, steering
