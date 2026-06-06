"""BeamNG telemetry polling and MPC→actuator mapping for live closed-loop runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from sindympc.data.telemetry import HEADING_VELOCITY_MIN_SPEED

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

LongitudinalMode = Literal["throttle", "brake", "coast"]
ActuatorProfile = Literal["responsive", "smooth"]


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
    """Convert desired speed [m/s] to throttle/brake with hysteresis (no bang-bang)."""

    dt: float
    kp: float = 0.35
    ki: float = 0.12
    kd: float = 0.04
    windup_limit: float = 0.8
    max_delta: float = 0.10
    max_throttle_delta: float | None = None
    max_brake_delta: float | None = None
    max_throttle: float = 0.65
    max_brake: float = 0.55
    throttle_on_err_mps: float = 0.20
    brake_on_err_mps: float = 0.20
    coast_damp: float = 0.35

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_err: float = field(default=0.0, init=False, repr=False)
    _prev_throttle: float = field(default=0.0, init=False, repr=False)
    _prev_brake: float = field(default=0.0, init=False, repr=False)
    _mode: LongitudinalMode = field(default="coast", init=False, repr=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._mode = "coast"

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

        if desired_speed < 0.08:
            self.reset()
            brake = float(np.clip(cur_speed * 0.25, 0.0, self.max_brake))
            return 0.0, brake

        err = desired_speed - cur_speed
        self._integral += err * self.dt
        self._integral = float(
            np.clip(self._integral, -self.windup_limit, self.windup_limit)
        )
        derivative = (err - self._prev_err) / self.dt if self.dt > 0 else 0.0
        self._prev_err = err

        output = self.kp * err + self.ki * self._integral + self.kd * derivative

        if err > self.throttle_on_err_mps:
            self._mode = "throttle"
        elif err < -self.brake_on_err_mps:
            self._mode = "brake"
        elif abs(err) < 0.08:
            self._mode = "coast"

        raw_throttle = float(np.clip(max(0.0, output), 0.0, self.max_throttle))
        raw_brake = float(np.clip(max(0.0, -output), 0.0, self.max_brake))

        if self._mode == "throttle":
            raw_brake *= self.coast_damp
        elif self._mode == "brake":
            raw_throttle *= self.coast_damp
        else:
            raw_throttle *= self.coast_damp
            raw_brake *= self.coast_damp

        throttle = float(
            np.clip(
                raw_throttle,
                self._prev_throttle - self._throttle_delta(),
                self._prev_throttle + self._throttle_delta(),
            )
        )
        brake = float(
            np.clip(
                raw_brake,
                self._prev_brake - self._brake_delta(),
                self._prev_brake + self._brake_delta(),
            )
        )
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        self._prev_throttle = throttle
        self._prev_brake = brake
        return throttle, brake

    def _throttle_delta(self) -> float:
        return self.max_throttle_delta if self.max_throttle_delta is not None else self.max_delta

    def _brake_delta(self) -> float:
        return self.max_brake_delta if self.max_brake_delta is not None else self.max_delta


@dataclass
class BeamNGActuatorMapper:
    """Smooth MPC/MPPI ``[a, delta]`` → BeamNG throttle, brake, steering.

    Use :meth:`for_controller` — MPPI needs heavy actuator smoothing; SQP-MPC
    needs a more responsive profile or steering lags and saturates.
    """

    dt: float
    max_acceleration: float
    max_steering_angle: float
    speed_horizon_s: float = 0.7
    desired_speed_alpha: float = 0.45
    steer_max_delta: float = 0.10
    control_smooth_alpha: float = 0.55
    max_speed_mps: float = 30.0
    min_creep_speed_mps: float = 1.2
    goal_position: tuple[float, float] | None = None
    goal_slow_radius_m: float = 30.0
    full_speed_goal_radius_m: float = 30.0
    target_speed_mps: float = 5.0
    sqp_cruise_governor: bool = False
    decel_feedforward_gain: float = 0.35
    throttle_assist_gain: float = 0.45
    speed_pid: SpeedPID = field(default_factory=lambda: SpeedPID(dt=0.1))

    _prev_desired_speed: float = field(default=0.0, init=False, repr=False)
    _prev_steering: float = field(default=0.0, init=False, repr=False)
    _prev_accel: float = field(default=0.0, init=False, repr=False)
    _prev_delta: float = field(default=0.0, init=False, repr=False)
    _controls_initialized: bool = field(default=False, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    @classmethod
    def for_controller(
        cls,
        controller: str,
        *,
        dt: float,
        max_acceleration: float,
        max_steering_angle: float,
        plant: Literal["bicycle", "sindy"] = "bicycle",
        profile: ActuatorProfile = "responsive",
        target_speed_mps: float = 5.0,
        goal_position: tuple[float, float] | None = None,
        goal_slow_radius_m: float = 30.0,
        full_speed_goal_radius_m: float = 30.0,
        sqp_cruise_governor: bool = False,
    ) -> BeamNGActuatorMapper:
        """Presets tuned for BeamNG closed-loop (not JAX open-loop sim)."""
        key = controller.strip().lower()
        is_sqp = key in ("sqp", "mpc")
        is_smooth = profile == "smooth"

        if is_sqp:
            # SQP: responsive throttle; less control smoothing so small ``a`` reaches PID.
            pid = SpeedPID(
                dt=dt,
                kp=0.48,
                ki=0.16,
                max_throttle=0.70,
                max_throttle_delta=0.16,
                max_brake_delta=0.07,
                brake_on_err_mps=0.28,
            )
            mapper = cls(
                dt=dt,
                max_acceleration=max_acceleration,
                max_steering_angle=max_steering_angle,
                speed_horizon_s=0.90,
                desired_speed_alpha=0.78 if not is_smooth else 0.55,
                steer_max_delta=0.30 if not is_smooth else 0.12,
                control_smooth_alpha=0.45 if not is_smooth else 0.50,
                throttle_assist_gain=0.55,
                decel_feedforward_gain=0.28,
                speed_pid=pid,
                goal_position=goal_position,
                goal_slow_radius_m=goal_slow_radius_m,
                full_speed_goal_radius_m=full_speed_goal_radius_m,
                target_speed_mps=target_speed_mps,
                sqp_cruise_governor=sqp_cruise_governor,
            )
        else:
            pid = SpeedPID(
                dt=dt,
                kp=0.38,
                ki=0.10,
                max_throttle=0.58,
                max_brake=0.38,
                max_throttle_delta=0.10 if is_smooth else 0.12,
                max_brake_delta=0.06,
                brake_on_err_mps=0.32,
            )
            mapper = cls(
                dt=dt,
                max_acceleration=max_acceleration,
                max_steering_angle=max_steering_angle,
                speed_horizon_s=0.75,
                desired_speed_alpha=0.45 if not is_smooth else 0.35,
                steer_max_delta=0.12,
                control_smooth_alpha=0.50 if not is_smooth else 0.40,
                decel_feedforward_gain=0.22,
                throttle_assist_gain=0.40,
                speed_pid=pid,
                goal_position=goal_position,
                goal_slow_radius_m=goal_slow_radius_m,
                full_speed_goal_radius_m=full_speed_goal_radius_m,
                target_speed_mps=target_speed_mps,
            )
        if plant == "sindy":
            mapper.min_creep_speed_mps = 1.2 if is_sqp else 1.8
            if is_sqp:
                mapper.speed_horizon_s = max(mapper.speed_horizon_s, 0.85)
            elif not is_sqp:
                mapper.speed_horizon_s *= 1.10
        mapper.max_speed_mps = max(mapper.max_speed_mps, target_speed_mps * 1.5)
        return mapper

    def __post_init__(self) -> None:
        if self.speed_pid.dt != self.dt:
            self.speed_pid.dt = self.dt

    def reset(self) -> None:
        self.speed_pid.reset()
        self._prev_desired_speed = 0.0
        self._prev_steering = 0.0
        self._prev_accel = 0.0
        self._prev_delta = 0.0
        self._controls_initialized = False
        self._initialized = False

    def _smooth_controls(self, accel: float, delta: float) -> Tuple[float, float]:
        if not self._controls_initialized:
            self._prev_accel = accel
            self._prev_delta = delta
            self._controls_initialized = True
            return accel, delta
        alpha = float(np.clip(self.control_smooth_alpha, 0.05, 1.0))
        accel = alpha * accel + (1.0 - alpha) * self._prev_accel
        delta = alpha * delta + (1.0 - alpha) * self._prev_delta
        self._prev_accel = accel
        self._prev_delta = delta
        return accel, delta

    def _smooth_desired_speed(self, raw: float, v_now: float) -> float:
        if not self._initialized:
            self._prev_desired_speed = raw
            self._initialized = True
            return raw
        alpha = float(np.clip(self.desired_speed_alpha, 0.05, 1.0))
        smoothed = alpha * raw + (1.0 - alpha) * self._prev_desired_speed
        max_step = max(0.4, abs(v_now) * 0.5) * self.dt * 8.0
        smoothed = float(
            np.clip(
                smoothed,
                self._prev_desired_speed - max_step,
                self._prev_desired_speed + max_step,
            )
        )
        self._prev_desired_speed = smoothed
        return smoothed

    def _goal_speed_scale(self, state: Sequence[float]) -> float:
        """Full cruise speed far from goal; ramp down inside ``goal_slow_radius_m``."""
        if self.goal_position is None:
            return 1.0
        dist = float(
            np.hypot(
                float(state[0]) - self.goal_position[0],
                float(state[1]) - self.goal_position[1],
            )
        )
        if dist >= self.full_speed_goal_radius_m:
            return 1.0
        if dist <= self.goal_slow_radius_m:
            return float(np.clip(dist / max(self.goal_slow_radius_m, 1e-3), 0.15, 1.0))
        # Linear blend between full-speed radius and slow radius.
        span = max(self.full_speed_goal_radius_m - self.goal_slow_radius_m, 1e-3)
        return float(
            np.clip(
                0.15 + 0.85 * (dist - self.goal_slow_radius_m) / span,
                0.15,
                1.0,
            )
        )

    def map_control(
        self,
        control: Sequence[float],
        state: Sequence[float],
        vel: Sequence[float],
    ) -> Tuple[float, float, float]:
        """Return ``(throttle, brake, steering)`` for ``vehicle.control(...)``."""
        a, delta = self._smooth_controls(float(control[0]), float(control[1]))
        v = float(state[3])
        dist_goal = float("inf")
        if self.goal_position is not None:
            dist_goal = float(
                np.hypot(
                    float(state[0]) - self.goal_position[0],
                    float(state[1]) - self.goal_position[1],
                )
            )

        raw_target = float(
            np.clip(v + a * self.speed_horizon_s, 0.0, self.max_speed_mps)
        )
        # SQP + weak SINDy v̇: hold cruise when far from goal if MPC ``a`` is tiny.
        if (
            self.sqp_cruise_governor
            and self.target_speed_mps > 0.0
            and dist_goal > self.goal_slow_radius_m * 1.5
        ):
            speed_err = self.target_speed_mps - v
            if speed_err > 0.6:
                cruise = min(
                    self.target_speed_mps,
                    v + max(0.35 * speed_err, 0.4),
                )
                raw_target = max(raw_target, cruise)
        raw_target *= self._goal_speed_scale(state)
        if a > 0.02 and v < max(0.8, self.min_creep_speed_mps):
            raw_target = max(raw_target, self.min_creep_speed_mps)

        desired_speed = self._smooth_desired_speed(raw_target, v)
        throttle, brake = self.speed_pid(desired_speed, vel)

        if a > 0.02 and throttle < 0.22 and brake < 0.05:
            assist = float(
                np.clip(
                    0.18 + self.throttle_assist_gain * (a / max(self.max_acceleration, 1e-3)),
                    0.12,
                    self.speed_pid.max_throttle,
                )
            )
            if dist_goal > self.goal_slow_radius_m:
                assist = max(assist, 0.20 + 0.25 * (a / max(self.max_acceleration, 1e-3)))
            throttle = max(throttle, min(assist, self.speed_pid.max_throttle))
        elif (
            self.sqp_cruise_governor
            and dist_goal > self.goal_slow_radius_m
            and v < self.target_speed_mps * 0.8
            and throttle < 0.25
            and brake < 0.05
        ):
            assist = float(
                np.clip(
                    0.22 + 0.35 * ((self.target_speed_mps - v) / max(self.target_speed_mps, 1e-3)),
                    0.18,
                    self.speed_pid.max_throttle,
                )
            )
            throttle = max(throttle, assist)
        elif a < -0.10 and (dist_goal > 6.0 or a < -0.30):
            decel = float(
                np.clip(
                    -a / max(self.max_acceleration, 1e-3) * self.decel_feedforward_gain,
                    0.03,
                    self.speed_pid.max_brake * 0.75,
                )
            )
            brake = max(brake, decel)
            throttle = min(throttle, throttle * 0.65)

        steer_norm = float(np.clip(delta / max(self.max_steering_angle, 1e-6), -1.0, 1.0))
        steering = -steer_norm
        steering = float(
            np.clip(
                steering,
                self._prev_steering - self.steer_max_delta,
                self._prev_steering + self.steer_max_delta,
            )
        )
        self._prev_steering = steering
        return throttle, brake, steering


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
    speed_horizon_s: float = 0.7,
    min_creep_speed_mps: float = 1.2,
    mapper: BeamNGActuatorMapper | None = None,
) -> Tuple[float, float, float]:
    """Map MPC/MPPI ``[a, delta]`` to BeamNG actuators (prefer :class:`BeamNGActuatorMapper`)."""
    if mapper is not None:
        return mapper.map_control(control, state, vel)

    m = BeamNGActuatorMapper(
        dt=dt,
        max_acceleration=max_acceleration,
        max_steering_angle=max_steering_angle,
        speed_horizon_s=speed_horizon_s,
        max_speed_mps=max_speed_mps,
        min_creep_speed_mps=min_creep_speed_mps,
        speed_pid=speed_pid,
    )
    return m.map_control(control, state, vel)


def open_beamng(bng, *, launch: bool, connect_retries: int = 30) -> None:
    """Connect to BeamNG.tech, optionally launching a new process.

    With ``launch=False`` (attach / ``--no-launch``), retries the socket
    connection because a manually started simulator can take several seconds
    to open its tech port.
    """
    from beamngpy.connection import Connection

    if launch:
        bng.open(launch=True)
        return

    bng.connection = Connection(bng.host, bng.port)
    connected = bng.connection.connect_to_beamng(
        tries=connect_retries,
        log_tries=True,
    )
    if not connected:
        cmd = bng.get_launch_arguments()
        raise SystemExit(
            f"[Error] Could not connect to BeamNG at {bng.host}:{bng.port} "
            "(attach / --no-launch mode).\n\n"
            "Start BeamNG.tech with the tech interface enabled, e.g.:\n\n"
            f"  {cmd}\n\n"
            "Or omit --no-launch to let the script launch BeamNG."
        )
    bng.logger.info("BeamNGpy connected to existing BeamNG instance.")
    bng._load_system_info()


def release_beamng(bng, *, keep_alive: bool) -> None:
    """Disconnect without quitting BeamNG when ``keep_alive`` is True."""
    try:
        bng.ui.show_hud()
    except Exception:
        pass
    if keep_alive:
        bng.disconnect()
    else:
        bng.close()
