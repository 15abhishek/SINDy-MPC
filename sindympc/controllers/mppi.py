"""MPPI (Model Predictive Path Integral) local controller.

Samples control sequences, rolls them forward with a JAX ``rk4_step`` plant,
scores quadratic tracking + optional obstacle penalties, and returns the
softmax-weighted mean of the first control (same interface as :class:`SQPMPC`).

Adapted from ``HybridVLA/hybridvla/controller/mppi.py`` for the SINDy-MPC
state ``[x, y, theta, v]`` and control ``[a, delta]``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sindympc.controllers.path_reference import waypoints_to_reference
from sindympc.controllers.sqp_mpc import CONTROL_DIM
from sindympc.dynamics.bicycle_model import rk4_step as bicycle_rk4_step

ObstacleDisk = Tuple[float, float, float]


class MPPIController:
    """Path-integral MPC with control space ``[acceleration, steering]`` [rad]."""

    def __init__(
        self,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        H: int,
        dt: float,
        max_acceleration: float,
        max_steering_angle: float,
        *,
        QN: Optional[jnp.ndarray] = None,
        num_samples: int = 256,
        lambda_: float = 1.0,
        noise_accel: float = 0.35,
        noise_steer: float = 0.12,
        target_speed: float = 2.0,
        wheel_base: float = 2.5,
        goal_position: Optional[Tuple[float, float]] = None,
        rk4_step_fn: Optional[
            Callable[[jnp.ndarray, jnp.ndarray, float, float], jnp.ndarray]
        ] = None,
        obstacle_list: Optional[Sequence[ObstacleDisk]] = None,
        obstacle_margin: float = 0.35,
        w_obstacle: float = 500.0,
        w_cross_track: float = 30.0,
        w_goal_xy: float = 0.0,
        w_goal_v: float = 0.0,
        goal_slow_radius: float = 20.0,
        R_delta: Optional[jnp.ndarray] = None,
        seed: int = 0,
    ) -> None:
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.QN = np.asarray(QN if QN is not None else Q, dtype=np.float64)
        self.H = H
        self.dt = dt
        self.max_acceleration = max_acceleration
        self.max_steering_angle = max_steering_angle
        self.K = num_samples
        self.lambda_ = lambda_
        self.noise_accel = noise_accel
        self.noise_steer = noise_steer
        self.target_speed = target_speed
        self.wheel_base = wheel_base
        self.goal_position = goal_position
        self.rk4_step_fn = rk4_step_fn or bicycle_rk4_step
        self.obstacle_list = list(obstacle_list or [])
        self.obstacle_margin = obstacle_margin
        self.w_obstacle = w_obstacle
        self.w_cross_track = w_cross_track
        self.w_goal_xy = w_goal_xy
        self.w_goal_v = w_goal_v
        self.goal_slow_radius = goal_slow_radius
        self._waypoints_xy: Optional[np.ndarray] = None
        self._R_delta = (
            np.asarray(R_delta, dtype=np.float64)
            if R_delta is not None
            else np.diag([0.05, 2.0])
        )

        self._rng = np.random.default_rng(seed)
        self._mean_controls = np.zeros((H, CONTROL_DIM), dtype=np.float64)
        self._last_arc_s: Optional[float] = None
        self.last_arc_s: Optional[float] = None
        self._waypoints_xy: Optional[np.ndarray] = None
        self._rollout_fn = self._build_rollout_fn()

    def reset_path_tracking(self) -> None:
        """Reset path progress and control warm-start."""
        self._mean_controls = np.zeros((self.H, CONTROL_DIM), dtype=np.float64)
        self._last_arc_s = None
        self.last_arc_s = None
        self._waypoints_xy = None

    def _build_rollout_fn(self):
        rk4 = self.rk4_step_fn
        dt = self.dt

        def step(state: jnp.ndarray, control: jnp.ndarray, wb: float):
            next_state = rk4(state, control, wb, dt)
            return next_state, next_state

        def rollout(initial: jnp.ndarray, controls: jnp.ndarray, wb: float):
            _, traj = jax.lax.scan(
                lambda s, u: step(s, u, wb), initial, controls
            )
            return jnp.concatenate([initial[None, :], traj], axis=0)

        return jax.jit(jax.vmap(rollout, in_axes=(None, 0, None)))

    def _sample_controls(self) -> np.ndarray:
        """Shape (K, H, 2) control sequences around the warm-started mean."""
        eps = self._rng.normal(
            0.0,
            [self.noise_accel, self.noise_steer],
            size=(self.K, self.H, CONTROL_DIM),
        )
        seqs = self._mean_controls[None, :, :] + eps
        seqs[..., 0] = np.clip(seqs[..., 0], -self.max_acceleration, self.max_acceleration)
        seqs[..., 1] = np.clip(seqs[..., 1], -self.max_steering_angle, self.max_steering_angle)
        return seqs

    def _stage_state_cost(
        self,
        state: np.ndarray,
        ref: np.ndarray,
    ) -> float:
        err = state - ref
        err[2] = np.arctan2(np.sin(err[2]), np.cos(err[2]))
        return float(err @ self.Q @ err)

    def _obstacle_cost(self, x: float, y: float) -> float:
        if not self.obstacle_list or self.w_obstacle <= 0.0:
            return 0.0
        total = 0.0
        for ox, oy, radius in self.obstacle_list:
            dist = float(np.hypot(x - ox, y - oy))
            clearance = dist - radius - self.obstacle_margin
            if clearance < 0.0:
                total += clearance * clearance
        return self.w_obstacle * total

    def _terminal_state_cost(self, state: np.ndarray, ref: np.ndarray) -> float:
        err = state - ref
        err[2] = np.arctan2(np.sin(err[2]), np.cos(err[2]))
        return float(err @ self.QN @ err)

    def _rollout_goal_cost(self, trajectory: np.ndarray) -> float:
        """Terminal goal XY + horizon speed penalty (discourage coasting past goal)."""
        if self.goal_position is None:
            return 0.0
        gx, gy = self.goal_position
        last = trajectory[-1]
        dx = float(last[0] - gx)
        dy = float(last[1] - gy)
        terminal_xy = self.w_goal_xy * (dx * dx + dy * dy)
        speed_sum = float(np.sum(trajectory[:, 3] ** 2))
        return terminal_xy + self.w_goal_v * speed_sum

    def _rollout_path_deviation_cost(self, trajectory: np.ndarray) -> float:
        """Sum of nearest-waypoint distances along a rollout (fast path-following proxy)."""
        if self._waypoints_xy is None or self.w_cross_track <= 0.0:
            return 0.0
        xy = trajectory[:, :2]
        dx = xy[:, 0:1] - self._waypoints_xy[None, :, 0]
        dy = xy[:, 1:2] - self._waypoints_xy[None, :, 1]
        nearest = np.min(np.hypot(dx, dy), axis=1)
        return self.w_cross_track * float(np.sum(nearest * nearest))

    def _evaluate_rollouts(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Per-sample total cost, shape (K,)."""
        K = trajectories.shape[0]
        costs = np.zeros(K, dtype=np.float64)

        for k in range(K):
            total = 0.0
            for t in range(self.H):
                total += self._stage_state_cost(trajectories[k, t], reference[t])
                total += float(controls[k, t] @ self.R @ controls[k, t])
                total += self._obstacle_cost(
                    float(trajectories[k, t, 0]),
                    float(trajectories[k, t, 1]),
                )
                if t > 0:
                    du = controls[k, t] - controls[k, t - 1]
                    total += float(du @ self._R_delta @ du)

            total += self._terminal_state_cost(trajectories[k, self.H], reference[self.H])
            total += self._rollout_path_deviation_cost(trajectories[k])
            total += self._rollout_goal_cost(trajectories[k])
            costs[k] = total

        return costs

    @staticmethod
    def _softmax_weights(costs: np.ndarray, lambda_: float) -> np.ndarray:
        shifted = costs - costs.min()
        log_w = -shifted / lambda_
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()

    def _shift_warmstart(self) -> None:
        self._mean_controls = np.roll(self._mean_controls, -1, axis=0)
        self._mean_controls[-1] = self._mean_controls[-2]

    def compute_action(
        self,
        current_state: jnp.ndarray,
        rrt_waypoints: List[Tuple[float, float]],
        wheel_base: Optional[float] = None,
    ) -> jnp.ndarray:
        """Sample rollouts, weight by cost, return first control ``[a, delta]``."""
        wb = float(wheel_base if wheel_base is not None else self.wheel_base)
        if rrt_waypoints:
            self._waypoints_xy = np.array(rrt_waypoints, dtype=np.float64)

        reference, self._last_arc_s = waypoints_to_reference(
            current_state,
            rrt_waypoints,
            horizon=self.H,
            dt=self.dt,
            target_speed=self.target_speed,
            goal_position=self.goal_position,
            last_arc_s=self._last_arc_s,
            goal_slow_radius=self.goal_slow_radius,
        )
        self.last_arc_s = self._last_arc_s
        ref_np = np.asarray(reference, dtype=np.float64)

        control_seqs = self._sample_controls()
        traj_jax = self._rollout_fn(
            current_state,
            jnp.asarray(control_seqs),
            wb,
        )
        trajectories = np.asarray(traj_jax, dtype=np.float64)

        costs = self._evaluate_rollouts(trajectories, control_seqs, ref_np)
        weights = self._softmax_weights(costs, self.lambda_)

        self._mean_controls = np.tensordot(weights, control_seqs, axes=(0, 0))
        u0 = jnp.array(self._mean_controls[0])
        self._shift_warmstart()

        if self.goal_position is not None:
            dist = float(
                jnp.hypot(
                    current_state[0] - self.goal_position[0],
                    current_state[1] - self.goal_position[1],
                )
            )
            v_now = float(current_state[3])
            # Tiered braking (SQP-style); SINDy needs earlier / stronger decel than bicycle.
            if dist < 8.0 and v_now > 0.5:
                scale = min(1.0, v_now / max(self.target_speed, 0.5))
                u0 = jnp.array(
                    [min(float(u0[0]), -scale * self.max_acceleration), float(u0[1])]
                )
            if dist < 2.5 and v_now > 0.12:
                u0 = jnp.array([-self.max_acceleration, 0.0])
            if dist < 1.0 and v_now > 0.05:
                u0 = jnp.array([-self.max_acceleration, 0.0])

        return u0
