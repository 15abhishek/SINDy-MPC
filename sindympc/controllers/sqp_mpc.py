"""Sequential Quadratic Programming Model Predictive Controller (SQP-MPC).

Structure mirrors ``lmc-exercises-private/homework_solutions/05_rocket_trajopt.ipynb``:
  - module-level ``linearize_dynamics`` with affine term C
  - CVXPY Variables + Parameters built once in ``_setup_cvxpy_problem``
  - SQP outer loop updates nominal trajectory, re-linearizes, and re-solves
"""

from __future__ import annotations

import functools
from typing import Callable, List, Optional, Tuple

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

from sindympc.dynamics.bicycle_model import rk4_step

# ---------------------------------------------------------------------------
# Dimensions (bicycle model)
# ---------------------------------------------------------------------------
STATE_DIM = 4  # [x, y, theta, v]
CONTROL_DIM = 2  # [a, delta]


# ---------------------------------------------------------------------------
# Linearization (homework 05 pattern)
# ---------------------------------------------------------------------------
def linearize_dynamics(
    dynamics: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    state: jnp.ndarray,
    control: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Affine linearization of discrete dynamics about (state, control).

    For f(x, u) = rk4_step(x, u, ...):
        f(x, u) ≈ A @ x + B @ u + C

    Args:
        dynamics: Function (state, control) -> next_state.
        state: Nominal state, shape (STATE_DIM,).
        control: Nominal control, shape (CONTROL_DIM,).

    Returns:
        A: State Jacobian, shape (STATE_DIM, STATE_DIM).
        B: Control Jacobian, shape (STATE_DIM, CONTROL_DIM).
        C: Affine offset, shape (STATE_DIM,).
    """
    A, B = jax.jacobian(dynamics, argnums=(0, 1))(state, control)
    C = dynamics(state, control) - A @ state - B @ control
    return A, B, C


def circular_obstacle_constraint(
    state: jnp.ndarray,
    obs_center: jnp.ndarray,
    obs_radius: float,
) -> jnp.ndarray:
    """Obstacle constraint (homework 05): feasible when value <= 0.

    Returns r^2 - ||p - c||^2 so G @ x + h <= 0 keeps the position outside the disk.
    """
    x, y = state[0], state[1]
    dist_sq = (x - obs_center[0]) ** 2 + (y - obs_center[1]) ** 2
    return obs_radius**2 - dist_sq


class SQPMPC:
    """SQP-MPC for tracking RRT waypoints with a bicycle model.

    Follows the rocket trajopt notebook workflow:
      1. Build a CVXPY problem once (decision vars + parameters).
      2. Each SQP iteration: linearize along nominal (x̄, ū), update parameters, solve QP.
      3. Use the converged control sequence; apply u₀ receding-horizon style.

    Args:
        Q: Stage state tracking weight, shape (STATE_DIM, STATE_DIM).
        R: Stage control penalty, shape (CONTROL_DIM, CONTROL_DIM).
        H: Prediction horizon (number of control stages).
        dt: Discrete time step [s].
        max_acceleration: |a| upper bound [m/s^2].
        max_steering_angle: |delta| upper bound [rad].
        Q_trp: Trust-region penalty on state deviation from nominal.
        R_trp: Trust-region penalty on control deviation from nominal.
        QN: Terminal state tracking weight, shape (STATE_DIM, STATE_DIM).
        wheel_base: Bicycle wheelbase L [m].
        sqp_max_iter: Maximum SQP outer iterations per ``compute_action`` call.
        sqp_tol: Convergence threshold on ||U - U_prev||.
        target_speed: Cruise speed for horizon lookahead [m/s] in reference generation.
        obstacle_list: Circular obstacles as (x, y, radius); enables linearized constraints.
        obstacle_margin: Extra clearance added to each obstacle radius [m].
        goal_position: Optional (x, y) for terminal reference blending near path end.
    """

    def __init__(
        self,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        H: int,
        dt: float,
        max_acceleration: float,
        max_steering_angle: float,
        Q_trp: Optional[jnp.ndarray] = None,
        R_trp: Optional[jnp.ndarray] = None,
        QN: Optional[jnp.ndarray] = None,
        wheel_base: float = 2.5,
        sqp_max_iter: int = 2,
        sqp_tol: float = 1e-3,
        solver: str = "OSQP",
        target_speed: float = 2.0,
        obstacle_list: Optional[List[Tuple[float, float, float]]] = None,
        obstacle_margin: float = 0.2,
        goal_position: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.Q = Q
        self.R = R
        self.H = H
        self.dt = dt
        self.max_acceleration = max_acceleration
        self.max_steering_angle = max_steering_angle
        self.wheel_base = wheel_base
        self.sqp_max_iter = sqp_max_iter
        self.sqp_tol = sqp_tol
        self.solver = solver
        self.target_speed = target_speed
        self.obstacle_list = obstacle_list or []
        self.obstacle_margin = obstacle_margin
        self.goal_position = goal_position
        self.n_obs = len(self.obstacle_list)

        # Warm-start control sequence across MPC calls (receding horizon)
        self._last_controls: Optional[jnp.ndarray] = None
        # Furthest arc-length progress along the path (prevents backward reference jumps)
        self._last_arc_s: Optional[float] = None

        # Trust-region and terminal weights (homework 05 defaults if omitted)
        self.Q_trp = (
            Q_trp if Q_trp is not None else 0.01 * jnp.eye(STATE_DIM)
        )
        self.R_trp = (
            R_trp if R_trp is not None else 0.01 * jnp.eye(CONTROL_DIM)
        )
        self.QN = QN if QN is not None else Q

        self._u_min = np.array([-max_acceleration, -max_steering_angle])
        self._u_max = np.array([max_acceleration, max_steering_angle])

        # CVXPY objects (populated in _setup_cvxpy_problem)
        self.states: Optional[cp.Variable] = None
        self.controls: Optional[cp.Variable] = None
        self.problem: Optional[cp.Problem] = None

        self._setup_cvxpy_problem()

    def reset_path_tracking(self) -> None:
        """Reset path progress and control warm-start (call at start of a new sim)."""
        self._last_controls = None
        self._last_arc_s = None

    @staticmethod
    def _project_onto_path(
        pos: np.ndarray,
        waypoints: np.ndarray,
        arc_length: np.ndarray,
    ) -> float:
        """Return arc-length coordinate of the closest point on the path polyline."""
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

    @staticmethod
    def _heading_along_path(
        s: float,
        arc_length: np.ndarray,
        waypoints: np.ndarray,
        eps: float = 0.2,
    ) -> float:
        """Path tangent heading at arc-length ``s`` via a small forward difference."""
        s_lo = max(float(arc_length[0]), s - eps)
        s_hi = min(float(arc_length[-1]), s + eps)
        if s_hi - s_lo < 1e-6:
            s_hi = min(float(arc_length[-1]), s_lo + eps)

        x_lo = np.interp(s_lo, arc_length, waypoints[:, 0])
        y_lo = np.interp(s_lo, arc_length, waypoints[:, 1])
        x_hi = np.interp(s_hi, arc_length, waypoints[:, 0])
        y_hi = np.interp(s_hi, arc_length, waypoints[:, 1])
        return float(np.arctan2(y_hi - y_lo, x_hi - x_lo))

    def _solve_problem(self) -> bool:
        """Solve the CVXPY QP using the configured backend.

        Returns:
            True if a usable solution was returned, False otherwise.
        """
        assert self.problem is not None
        if self.solver.upper() == "OSQP":
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1e-3,
                eps_rel=1e-3,
                max_iter=8000,
                polish=False,
            )
        else:
            self.problem.solve(solver=cp.CLARABEL, verbose=False)

        ok = self.problem.status in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        )
        if self.controls is None or self.controls.value is None:
            ok = False
        return ok

    def _make_discrete_dynamics(
        self, wheel_base: float
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Return f(x, u) = rk4_step(x, u, wheel_base, dt) for JAX linearization."""

        def f(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
            return rk4_step(state, control, wheel_base, self.dt)

        return f

    def _setup_cvxpy_problem(self) -> None:
        """Build the CVXPY problem once (homework 05 cell structure).

        Decision variables:
            states  : (H + 1, STATE_DIM)
            controls: (H, CONTROL_DIM)

        Parameters (updated each SQP iteration):
            As, Bs, Cs              — time-varying linearized dynamics
            previous_states         — nominal state rollout x̄
            previous_controls       — nominal control sequence ū
            initial_state           — measured x₀
            reference_states        — MPC reference X_ref (homework 07 pattern)
        """
        H = self.H

        # --- Decision variables (rocket notebook) ---
        self.states = cp.Variable((H + 1, STATE_DIM), name="states")
        self.controls = cp.Variable((H, CONTROL_DIM), name="controls")

        # --- Time-varying linearization parameters ---
        self.As = cp.Parameter((H, STATE_DIM, STATE_DIM), name="As")
        self.Bs = cp.Parameter((H, STATE_DIM, CONTROL_DIM), name="Bs")
        self.Cs = cp.Parameter((H, STATE_DIM), name="Cs")

        # --- Nominal trajectory for trust-region penalties ---
        self.previous_states = cp.Parameter((H + 1, STATE_DIM), name="previous_states")
        self.previous_controls = cp.Parameter((H, CONTROL_DIM), name="previous_controls")

        # --- MPC reference and initial condition (07_mpc notebook) ---
        self.initial_state = cp.Parameter(STATE_DIM, name="initial_state")
        self.reference_states = cp.Parameter((H + 1, STATE_DIM), name="reference_states")

        # --- Control bounds ---
        self.u_min = cp.Constant(self._u_min, name="u_min")
        self.u_max = cp.Constant(self._u_max, name="u_max")

        # --- Cost matrices as constants ---
        Q = cp.Constant(np.array(self.Q), name="Q")
        R = cp.Constant(np.array(self.R), name="R")
        QN = cp.Constant(np.array(self.QN), name="QN")
        Q_trp = cp.Constant(np.array(self.Q_trp), name="Q_trp")
        R_trp = cp.Constant(np.array(self.R_trp), name="R_trp")

        # --- Objective and constraints (rocket notebook loop) ---
        objective = 0
        constraints = [self.states[0, :] == self.initial_state]

        for k in range(H):
            # Stage costs
            objective += cp.quad_form(self.controls[k, :], R)
            # Smooth steering / acceleration changes
            if k > 0:   # temporary, remove later if doesn't work
                Rd = cp.Constant(np.diag([0.1, 1.0]))
                objective += cp.quad_form(
                    self.controls[k, :] - self.controls[k - 1, :],
                    Rd,
                )
            objective += cp.quad_form(
                self.states[k, :] - self.reference_states[k, :], Q
            )
            # Trust-region penalties (keep iterate near linearization point)
            objective += cp.quad_form(
                self.states[k, :] - self.previous_states[k, :], Q_trp
            )
            objective += cp.quad_form(
                self.controls[k, :] - self.previous_controls[k, :], R_trp
            )

            # Affine dynamics: x_{k+1} = A_k x_k + B_k u_k + C_k
            constraints += [
                self.states[k + 1, :]
                == self.As[k] @ self.states[k, :]
                + self.Bs[k] @ self.controls[k, :]
                + self.Cs[k]
            ]

            # Actuation limits
            constraints += [self.controls[k, :] >= self.u_min]
            constraints += [self.controls[k, :] <= self.u_max]

        # Terminal cost
        objective += cp.quad_form(
            self.states[H, :] - self.reference_states[H, :], QN
        )

        # Linearized circular obstacle constraints (homework 05)
        self.G_obs: Optional[cp.Parameter] = None
        self.h_obs: Optional[cp.Parameter] = None
        if self.n_obs > 0:
            self.G_obs = cp.Parameter(
                (H + 1, self.n_obs, STATE_DIM), name="G_obs"
            )
            self.h_obs = cp.Parameter((H + 1, self.n_obs), name="h_obs")
            for k in range(H + 1):
                for o in range(self.n_obs):
                    constraints += [
                        self.G_obs[k, o, :] @ self.states[k, :]
                        + self.h_obs[k, o]
                        <= 0
                    ]

        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def _linearize_dynamics(
        self,
        nominal_state: jnp.ndarray,
        nominal_control: jnp.ndarray,
        wheel_base: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Linearize rk4_step about one (state, control) pair.

        Thin wrapper around the module-level ``linearize_dynamics``.
        """
        dynamics = self._make_discrete_dynamics(wheel_base)
        return linearize_dynamics(dynamics, nominal_state, nominal_control)

    def _rollout_nominal_trajectory(
        self,
        initial_state: jnp.ndarray,
        control_sequence: jnp.ndarray,
        wheel_base: float,
    ) -> jnp.ndarray:
        """Roll out nonlinear dynamics to obtain nominal states x̄.

        Args:
            initial_state: x₀, shape (STATE_DIM,).
            control_sequence: ū, shape (H, CONTROL_DIM).
            wheel_base: Vehicle wheelbase [m].

        Returns:
            Nominal states, shape (H + 1, STATE_DIM).
        """
        def step(state: jnp.ndarray, control: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            next_state = rk4_step(state, control, wheel_base, self.dt)
            return next_state, next_state

        _, trajectory = jax.lax.scan(step, initial_state, control_sequence)
        return jnp.concatenate([initial_state[None, :], trajectory], axis=0)

    def _linearize_along_nominal(
        self,
        nominal_states: jnp.ndarray,
        nominal_controls: jnp.ndarray,
        wheel_base: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Vmap ``linearize_dynamics`` over the nominal horizon (homework 05).

        Args:
            nominal_states: x̄, shape (H + 1, STATE_DIM).
            nominal_controls: ū, shape (H, CONTROL_DIM).
            wheel_base: Vehicle wheelbase [m].

        Returns:
            As: (H, STATE_DIM, STATE_DIM)
            Bs: (H, STATE_DIM, CONTROL_DIM)
            Cs: (H, STATE_DIM)
        """
        dynamics = self._make_discrete_dynamics(wheel_base)
        return jax.vmap(linearize_dynamics, in_axes=(None, 0, 0))(
            dynamics, nominal_states[:-1], nominal_controls
        )

    def _linearize_obstacle_constraints(
        self,
        nominal_states: jnp.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize stay-outside-disk constraints at each horizon stage."""
        G = np.zeros((self.H + 1, self.n_obs, STATE_DIM), dtype=np.float64)
        h = np.zeros((self.H + 1, self.n_obs), dtype=np.float64)

        for o, (ox, oy, radius) in enumerate(self.obstacle_list):
            r_eff = radius + self.obstacle_margin
            center = jnp.array([ox, oy], dtype=jnp.float64)

            constraint_fn = functools.partial(
                circular_obstacle_constraint,
                obs_center=center,
                obs_radius=r_eff,
            )
            Gs = jax.vmap(jax.jacobian(constraint_fn))(nominal_states)
            gs = jax.vmap(constraint_fn)(nominal_states)
            hs = gs - jnp.einsum("ij,ij->i", Gs, nominal_states)

            G[:, o, :] = np.array(Gs)
            h[:, o] = np.array(hs)

        return G, h

    def _update_problem_parameters(
        self,
        current_state: jnp.ndarray,
        reference_trajectory: jnp.ndarray,
        previous_states: jnp.ndarray,
        previous_controls: jnp.ndarray,
        wheel_base: float,
    ) -> None:
        """Assign CVXPY parameter values before each QP solve (homework 05/07).

        Steps:
            1. Linearize: As, Bs, Cs = vmap(linearize_dynamics)(x̄[:-1], ū)
            2. Set initial_state, reference_states, previous_* , As/Bs/Cs .value
        """
        As_values, Bs_values, Cs_values = self._linearize_along_nominal(
            previous_states, previous_controls, wheel_base
        )

        self.As.value = np.array(As_values)
        self.Bs.value = np.array(Bs_values)
        self.Cs.value = np.array(Cs_values)

        self.initial_state.value = np.array(current_state)
        self.reference_states.value = np.array(reference_trajectory)
        self.previous_states.value = np.array(previous_states)
        self.previous_controls.value = np.array(previous_controls)

        if self.n_obs > 0 and self.G_obs is not None and self.h_obs is not None:
            # Linearize about the RRT-based reference (collision-free), not the
            # rollout, which often cuts through obstacles when tracking is poor.
            # G_obs, h_obs = self._linearize_obstacle_constraints(reference_trajectory)         # original
            G_obs, h_obs = self._linearize_obstacle_constraints(previous_states)                # temporary, if doesn't work
            self.G_obs.value = G_obs
            self.h_obs.value = h_obs

    def _construct_qp_and_solve(
        self,
        current_state: jnp.ndarray,
        reference_trajectory: jnp.ndarray,
        initial_guess_u: jnp.ndarray,
        wheel_base: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Solve one SQP subproblem (single QP) for the current nominal iterate.

        Args:
            current_state: Measured x₀, shape (STATE_DIM,).
            reference_trajectory: X_ref, shape (H + 1, STATE_DIM).
            initial_guess_u: Nominal ū, shape (H, CONTROL_DIM).
            wheel_base: Vehicle wheelbase [m].

        Returns:
            optimal_states: (H + 1, STATE_DIM) from CVXPY solution.
            optimal_controls: (H, CONTROL_DIM) from CVXPY solution.
        """
        previous_states = self._rollout_nominal_trajectory(
            current_state, initial_guess_u, wheel_base
        )

        self._update_problem_parameters(
            current_state,
            reference_trajectory,
            previous_states,
            initial_guess_u,
            wheel_base,
        )

        assert self.problem is not None
        self._solve_problem()

        assert self.states is not None and self.controls is not None
        return jnp.array(self.states.value), jnp.array(self.controls.value)

    def _waypoints_to_reference(
        self,
        current_state: jnp.ndarray,
        rrt_waypoints: List[Tuple[float, float]],
    ) -> jnp.ndarray:
        """Build a receding-horizon reference by arc-length lookahead on the RRT path.

        Instead of compressing the full path into [0, 1], this tracks a short
        window ahead of the vehicle:
          1. Find the waypoint closest to the current (x, y).
          2. March forward along the path by ``target_speed * dt`` per stage.
          3. Interpolate (x, y) at those arc-length targets.

        Example: H=3, dt=0.1, v_target=2.0 → lookahead [0, 0.2, 0.4, 0.6] m.

        Args:
            current_state: Current [x, y, theta, v].
            rrt_waypoints: Planner waypoints as (x, y) tuples along the polyline.

        Returns:
            reference_trajectory, shape (H + 1, STATE_DIM).
        """
        waypoints = np.asarray(rrt_waypoints, dtype=np.float64)
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, 2)
        if waypoints.shape[0] == 1:
            waypoints = np.vstack([waypoints, waypoints])

        pos = np.array(
            [float(current_state[0]), float(current_state[1])], dtype=np.float64
        )

        seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        arc_length = np.zeros(waypoints.shape[0], dtype=np.float64)
        arc_length[1:] = np.cumsum(seg_lens)

        s_closest = self._project_onto_path(pos, waypoints, arc_length)
        if self._last_arc_s is None:
            s_base = s_closest
        else:
            # Never rewind the path index — critical when the car is off-track
            s_base = max(s_closest, self._last_arc_s)
        self._last_arc_s = s_base

        lookahead = np.arange(self.H + 1, dtype=np.float64) * self.dt * self.target_speed
        s_targets = np.clip(s_base + lookahead, arc_length[0], arc_length[-1])

        x_ref = np.interp(s_targets, arc_length, waypoints[:, 0])
        y_ref = np.interp(s_targets, arc_length, waypoints[:, 1])

        theta_ref = np.array(
            [
                self._heading_along_path(float(s), arc_length, waypoints)
                for s in s_targets
            ],
            dtype=np.float64,
        )

        v_now = float(current_state[3])
        dist_to_goal = float(
            np.hypot(pos[0] - self.goal_position[0], pos[1] - self.goal_position[1])
        ) if self.goal_position is not None else np.inf

        v_ref = np.empty(self.H + 1, dtype=np.float64)
        for k in range(self.H + 1):
            alpha = min(1.0, (k + 1) / max(self.H, 1))
            v_ref[k] = (1.0 - alpha) * v_now + alpha * self.target_speed

        # Blend references toward the goal in the last ~1.5 m of the path
        if self.goal_position is not None:
            s_max = float(arc_length[-1])
            gx, gy = self.goal_position
            goal_theta = self._heading_along_path(s_max, arc_length, waypoints)
            blend_len = 1.5
            for k in range(self.H + 1):
                blend = np.clip((s_targets[k] - (s_max - blend_len)) / blend_len, 0.0, 1.0)
                x_ref[k] = (1.0 - blend) * x_ref[k] + blend * gx
                y_ref[k] = (1.0 - blend) * y_ref[k] + blend * gy
                theta_ref[k] = (1.0 - blend) * theta_ref[k] + blend * goal_theta
                v_ref[k] = (1.0 - blend) * v_ref[k] + blend * self.target_speed

            # Geographic goal phase: slow down before overshooting
            if dist_to_goal < 3.0:
                slow = np.clip(dist_to_goal / 3.0, 0.0, 1.0)
                v_ref *= slow
            if dist_to_goal < 1.2:
                lock = np.clip((1.2 - dist_to_goal) / 1.2, 0.0, 1.0)
                x_ref = (1.0 - lock) * x_ref + lock * gx
                y_ref = (1.0 - lock) * y_ref + lock * gy
                theta_ref = (1.0 - lock) * theta_ref + lock * goal_theta
                v_ref = (1.0 - lock) * v_ref  # request stop at goal

        ref = np.stack([x_ref, y_ref, theta_ref, v_ref], axis=1)
        return jnp.array(ref)

    def compute_action(
        self,
        current_state: jnp.ndarray,
        rrt_waypoints: List[Tuple[float, float]],
        wheel_base: Optional[float] = None,
    ) -> jnp.ndarray:
        """MPC entry point: SQP loop and return first control u₀.

        Mirrors homework 05 SQP iteration:
            for i in range(sqp_max_iter):
                previous_states.value = ...
                As, Bs, Cs = vmap(linearize_dynamics)(...)
                problem.solve()
                previous_states, previous_controls <- solution
                if converged: break
            return controls[0]

        Args:
            current_state: [x, y, theta, v].
            rrt_waypoints: High-level planner waypoints.
            wheel_base: Override default wheelbase if provided.

        Returns:
            u₀, shape (CONTROL_DIM,) — [a, delta].
        """
        wb = wheel_base if wheel_base is not None else self.wheel_base

        reference_trajectory = self._waypoints_to_reference(
            current_state, rrt_waypoints
        )

        # Warm-start from previous MPC solution (shift horizon) or zeros
        if self._last_controls is not None:
            previous_controls = jnp.vstack(
                [self._last_controls[1:], self._last_controls[-1:]]
            )
        else:
            previous_controls = jnp.zeros((self.H, CONTROL_DIM))

        previous_states = self._rollout_nominal_trajectory(
            current_state, previous_controls, wb
        )

        for _ in range(self.sqp_max_iter):
            self._update_problem_parameters(
                current_state,
                reference_trajectory,
                previous_states,
                previous_controls,
                wb,
            )

            if not self._solve_problem():
                # QP infeasible / inaccurate — keep previous iterate, don't crash
                print("MPC solve failed, continuing with previous iterate")
                break

            assert self.states is not None and self.controls is not None
            new_controls = jnp.array(self.controls.value)
            print("u =", np.array(new_controls[0]))

            if float(jnp.linalg.norm(new_controls - previous_controls)) < self.sqp_tol:
                previous_controls = new_controls
                break

            previous_controls = new_controls
            # Re-linearize about the *nonlinear* rollout, not the QP state trajectory
            previous_states = self._rollout_nominal_trajectory(
                current_state, previous_controls, wb
            )

        self._last_controls = previous_controls
        u0 = previous_controls[0]

        # Emergency brake near goal if the QP still commands speed
        if self.goal_position is not None:
            dist = float(
                jnp.hypot(
                    current_state[0] - self.goal_position[0],
                    current_state[1] - self.goal_position[1],
                )
            )
            if dist < 1.0 and float(current_state[3]) > 0.2:
                u0 = jnp.array([-self.max_acceleration, 0.0])

        return u0
