"""Closed-loop simulation: RRT global plan + SQP-MPC tracking.

Pipeline:
    1. Plan once with RRT to obtain waypoint path.
    2. Simulate bicycle dynamics with receding-horizon SQP-MPC.
    3. Plot obstacles, RRT baseline path, and executed trajectory.
"""

from __future__ import annotations

import os
import time

# Force CPU JAX if CUDA plugins are installed but the driver is incompatible.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from types import ModuleType

from sindympc.config.beamng_defaults import (
    BEAMNG_DT,
    BEAMNG_GOAL_SLOW_RADIUS_M,
    BEAMNG_GOAL_THRESHOLD_M,
    BEAMNG_GOAL_XY,
    BEAMNG_INITIAL_SPEED_MPS,
    BEAMNG_RRT_STEP_SIZE,
    BEAMNG_SCENARIO_X_BOUNDS,
    BEAMNG_SCENARIO_Y_BOUNDS,
    BEAMNG_START_XY,
    BEAMNG_STATE_HIGH,
    BEAMNG_STATE_LOW,
    BEAMNG_TARGET_SPEED_MPS,
    BEAMNG_X_BOUNDS,
    BEAMNG_Y_BOUNDS,
)
from sindympc.controllers.sqp_mpc import CONTROL_DIM, STATE_DIM, SQPMPC
from sindympc.dynamics import DynamicsName, get_dynamics
from sindympc.planners.rrt import RRT
from sindympc.utils.dynamic_obstacles import DynamicObstacleField, MovingObstacle
from sindympc.utils.obstacles import (
    DEFAULT_MIN_DIST_FROM_POINTS,
    DEFAULT_NUM_OBSTACLES,
    DEFAULT_RADIUS_RANGE,
    nearest_obstacles,
    pad_obstacle_list,
    sample_random_obstacles,
)
from sindympc.utils.path_utils import densify_polyline, smooth_path_moving_average
from sindympc.utils.plot_utils import apply_scenario_axes
from sindympc.utils.sim_animation import (
    TRAIL_COLOR,
    TRAIL_LABEL,
    TRAIL_LINEWIDTH,
    AnimationConfig,
    animate_mpc_simulation,
)

# -----------------------------------------------------------------------------
# Dynamics plant: "bicycle" (kinematic) or "sindy" (data/sindy_model.npz)
# RRT is geometric only; this switch affects MPC + closed-loop simulation.
# -----------------------------------------------------------------------------
# DYNAMICS_MODEL: DynamicsName = "bicycle"
DYNAMICS_MODEL: DynamicsName = "sindy"
# MPC internal model — set equal to plant for pure SINDy vs pure bicycle comparison.
# Use mpc="bicycle" only as a deliberate ablation (predictor/plant mismatch).
MPC_DYNAMICS_MODEL: DynamicsName = DYNAMICS_MODEL

# -----------------------------------------------------------------------------
# Map / scenario — BeamNG world frame (matches data/trajectory.jsonl & SINDy training)
# -----------------------------------------------------------------------------

GOAL_POSITION: Tuple[float, float] = BEAMNG_GOAL_XY

_INIT_HEADING = float(
    np.arctan2(
        GOAL_POSITION[1] - BEAMNG_START_XY[1],
        GOAL_POSITION[0] - BEAMNG_START_XY[0],
    )
)
START_STATE = jnp.array(
    [
        BEAMNG_START_XY[0],
        BEAMNG_START_XY[1],
        _INIT_HEADING,
        BEAMNG_INITIAL_SPEED_MPS,
    ]
)  # [x, y, theta, v]

RRT_START: Tuple[float, float] = BEAMNG_START_XY
RRT_GOAL: Tuple[float, float] = GOAL_POSITION
RRT_BOUNDS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    BEAMNG_X_BOUNDS,
    BEAMNG_Y_BOUNDS,
)

OBSTACLE_LIST: List[Tuple[float, float, float]] = sample_random_obstacles(
    DEFAULT_NUM_OBSTACLES,
    BEAMNG_SCENARIO_X_BOUNDS,
    BEAMNG_SCENARIO_Y_BOUNDS,
    DEFAULT_RADIUS_RANGE,
    RRT_START,
    RRT_GOAL,
    min_dist_from_points=DEFAULT_MIN_DIST_FROM_POINTS,
)

# -----------------------------------------------------------------------------
# Dynamic obstacles (movers) — edit here; static ``OBSTACLE_LIST`` is unchanged.
# RRT uses static disks only. Movers appear in sim, MPC (optional), and GIF replay.
# -----------------------------------------------------------------------------
USE_DYNAMIC_OBSTACLES: bool = os.environ.get(
    "SINDYMPC_DYNAMIC_OBSTACLES", "1"
).lower() in ("1", "true", "yes")

# Each entry: (x0, y0, radius [m], vx [m/s], vy [m/s]) at t=0
DYNAMIC_OBSTACLE_FIELD: Optional[DynamicObstacleField] = (
    DynamicObstacleField(
        movers=[
            MovingObstacle(230.0, 25.0, 5.0, -0.5, 1.2),
            MovingObstacle(255.0, 55.0, 4.5, 0.8, -0.6),
            MovingObstacle(210.0, 70.0, 5.5, 1.0, 0.4),
        ]
    )
    if USE_DYNAMIC_OBSTACLES
    else None
)

# MPC tracks nearest dynamic disks only (keeps QP small; static handled by RRT).
MPC_DYNAMIC_OBSTACLE_SLOTS: int = (
    len(DYNAMIC_OBSTACLE_FIELD.movers)
    if DYNAMIC_OBSTACLE_FIELD is not None
    else 0
)

# RRT planner parameters (scaled for ~100+ m routes)
RRT_STEP_SIZE: float = BEAMNG_RRT_STEP_SIZE
RRT_MAX_ITER: int = 10000
PADDING_FRACTION: float = 0.05  # bounds already set explicitly

# Simulation parameters (dt matches telemetry used for SINDy)
DT: float = BEAMNG_DT
_MAX_ROUTE_M = float(
    np.hypot(
        BEAMNG_GOAL_XY[0] - BEAMNG_START_XY[0],
        BEAMNG_GOAL_XY[1] - BEAMNG_START_XY[1],
    )
)
MAX_SIM_STEPS: int = int(np.ceil(_MAX_ROUTE_M / (BEAMNG_TARGET_SPEED_MPS * DT))) + 150
GOAL_THRESHOLD: float = BEAMNG_GOAL_THRESHOLD_M

# Vehicle / MPC parameters
WHEEL_BASE: float = 2.5
MAX_ACCELERATION: float = 1.0
# Match SINDy training envelope (BEAMNG_CONTROL_HIGH[1] == 1.0 rad)
MAX_STEERING_ANGLE: float = 1.0
MPC_HORIZON: int = 25
PATH_SPACING_M: float = 3.0
# RRT avoids obstacles geometrically. Linearized obstacle QPs (15 disks × H stages)
# often yield inaccurate OSQP solutions and hurt both bicycle and SINDy tracking.
# Dynamic movers in MPC QP when enabled; static-only via SINDYMPC_MPC_OBSTACLES=1.
USE_MPC_OBSTACLE_CONSTRAINTS: bool = (
    (USE_DYNAMIC_OBSTACLES and MPC_DYNAMIC_OBSTACLE_SLOTS > 0)
    or os.environ.get("SINDYMPC_MPC_OBSTACLES", "").lower()
    in ("1", "true", "yes")
)
PROFILE_MPC: bool = os.environ.get("SINDYMPC_PROFILE_MPC", "").lower() in (
    "1",
    "true",
    "yes",
)
# Visualization: static plot, animation replay, or both
SHOW_STATIC_PLOT: bool = os.environ.get("SINDYMPC_STATIC_PLOT", "1").lower() in (
    "1",
    "true",
    "yes",
)
# GIF export only (no on-screen player — toolbar controls rarely work outside notebooks).
SHOW_ANIMATION: bool = os.environ.get("SINDYMPC_ANIMATE", "0").lower() in (
    "1",
    "true",
    "yes",
)
_DEFAULT_GIF = os.path.join("results", "mpc_replay.gif")
ANIMATION_SAVE_PATH: Optional[str] = (
    os.environ.get("SINDYMPC_ANIMATE_SAVE")
    or (_DEFAULT_GIF if SHOW_ANIMATION else None)
)
ANIMATION_INTERVAL_MS: int = int(os.environ.get("SINDYMPC_ANIMATE_INTERVAL_MS", "50"))
TARGET_SPEED: float = BEAMNG_TARGET_SPEED_MPS

# Cost matrices (tune as needed)
Q = jnp.diag(jnp.array([
    15.0,   # x
    15.0,   # y
    3.0,    # theta
    0.1,    # velocity
]))
_USE_SINDY_PLANT = DYNAMICS_MODEL == "sindy"
R_ACCEL = 0.05 if _USE_SINDY_PLANT else 0.03
R_STEER = 0.2 if _USE_SINDY_PLANT else 0.005
R = jnp.diag(jnp.array([R_ACCEL, R_STEER]))
R_DELTA = jnp.diag(
    jnp.array([0.08, 12.0] if _USE_SINDY_PLANT else [0.1, 1.0])
)
R_TRP = jnp.diag(jnp.array([0.05, 0.05])) if _USE_SINDY_PLANT else None
SQP_MAX_ITER = 4 if USE_DYNAMIC_OBSTACLES else 3
QN = jnp.diag(jnp.array([
    80.0,  # terminal x — pull onto goal at end of path
    80.0,  # terminal y
    8.0,
    1.0,
]))
OBSTACLE_MARGIN: float = 0.6 if USE_DYNAMIC_OBSTACLES else 0.35
GOAL_SLOW_RADIUS: float = BEAMNG_GOAL_SLOW_RADIUS_M

def world_obstacles_at_step(step: int) -> List[Tuple[float, float, float]]:
    """Static ``OBSTACLE_LIST`` plus dynamic disks at simulation step ``step``."""
    if DYNAMIC_OBSTACLE_FIELD is None:
        return list(OBSTACLE_LIST)
    return DYNAMIC_OBSTACLE_FIELD.combined_with_static(OBSTACLE_LIST, step, DT)


def mpc_dynamic_obstacles_at_step(step: int) -> List[Tuple[float, float, float]]:
    """Dynamic mover disks at simulation step ``step`` (padded for MPC slots)."""
    if DYNAMIC_OBSTACLE_FIELD is None or MPC_DYNAMIC_OBSTACLE_SLOTS <= 0:
        return []
    pool = DYNAMIC_OBSTACLE_FIELD.disks_at_step(step, DT)
    return pad_obstacle_list(pool, MPC_DYNAMIC_OBSTACLE_SLOTS)


def mpc_obstacle_horizon(sim_step: int) -> List[List[Tuple[float, float, float]]]:
    """Predicted dynamic disk layout for each MPC stage (sim_step .. sim_step+H)."""
    return [
        mpc_dynamic_obstacles_at_step(sim_step + k) for k in range(MPC_HORIZON + 1)
    ]


def run_rrt_planner() -> Tuple[Optional[List[Tuple[float, float]]], RRT]:
    """Run RRT once and return waypoints plus the planner instance."""
    planner = RRT(
        start=RRT_START,
        goal=RRT_GOAL,
        bounds=RRT_BOUNDS,
        padding_fraction=PADDING_FRACTION,
        obstacle_list=OBSTACLE_LIST,
        step_size=RRT_STEP_SIZE,
        max_iter=RRT_MAX_ITER,
    )
    path = planner.plan()
    return path, planner


def build_mpc_controller(
    mpc_dynamics: ModuleType,
    initial_state: jnp.ndarray,
) -> SQPMPC:
    """Construct SQP-MPC; ``mpc_dynamics`` is the model used inside the optimizer."""
    use_fd = False
    if USE_MPC_OBSTACLE_CONSTRAINTS and DYNAMIC_OBSTACLE_FIELD is not None:
        mpc_obstacles = mpc_dynamic_obstacles_at_step(0)
    elif USE_MPC_OBSTACLE_CONSTRAINTS:
        mpc_obstacles = list(OBSTACLE_LIST)
    else:
        mpc_obstacles = []
    kwargs: dict = dict(
        Q=Q,
        R=R,
        H=MPC_HORIZON,
        dt=DT,
        max_acceleration=MAX_ACCELERATION,
        max_steering_angle=MAX_STEERING_ANGLE,
        wheel_base=WHEEL_BASE,
        sqp_max_iter=SQP_MAX_ITER,
        solver="OSQP",
        target_speed=TARGET_SPEED,
        QN=QN,
        obstacle_list=mpc_obstacles,
        obstacle_margin=OBSTACLE_MARGIN,
        goal_position=GOAL_POSITION,
        rk4_step_fn=mpc_dynamics.rk4_step,
        use_fd_linearization=use_fd,
        R_delta=R_DELTA,
        verbose=False,
    )
    if R_TRP is not None:
        kwargs["R_trp"] = R_TRP
    return SQPMPC(**kwargs)


def plot_simulation_results(
    static_obstacles: List[Tuple[float, float, float]],
    rrt_path: Optional[List[Tuple[float, float]]],
    executed_trajectory: np.ndarray,
    goal_position: Tuple[float, float],
    start_state: jnp.ndarray,
    *,
    dynamic_obstacles: Optional[List[Tuple[float, float, float]]] = None,
) -> None:
    """Plot static + optional dynamic obstacles, RRT path, and MPC trajectory."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for ox, oy, radius in static_obstacles:
        ax.add_patch(Circle((ox, oy), radius, color="dimgray", alpha=0.6))
    if dynamic_obstacles:
        for ox, oy, radius in dynamic_obstacles:
            ax.add_patch(Circle((ox, oy), radius, color="darkorange", alpha=0.65))

    if rrt_path is not None and len(rrt_path) > 0:
        rrt_xs, rrt_ys = zip(*rrt_path)
        ax.plot(rrt_xs, rrt_ys, "g--", linewidth=2, label="RRT path")

    if executed_trajectory.size > 0:
        ax.plot(
            executed_trajectory[:, 0],
            executed_trajectory[:, 1],
            color=TRAIL_COLOR,
            linestyle="-",
            linewidth=TRAIL_LINEWIDTH,
            label=TRAIL_LABEL,
        )

    ax.plot(
        float(start_state[0]),
        float(start_state[1]),
        "go",
        markersize=10,
        label="Start",
        zorder=5,
    )
    ax.plot(
        goal_position[0],
        goal_position[1],
        "r*",
        markersize=14,
        label="Goal",
        zorder=5,
    )

    extra_xy = [goal_position, (float(start_state[0]), float(start_state[1]))]
    if rrt_path:
        extra_xy.extend(rrt_path)
    apply_scenario_axes(
        ax,
        trajectory_xy=executed_trajectory[:, :2] if executed_trajectory.size else None,
        extra_xy=extra_xy,
    )
    title = f"RRT + SQP-MPC (plant={DYNAMICS_MODEL}"
    if MPC_DYNAMICS_MODEL != DYNAMICS_MODEL:
        title += f", mpc={MPC_DYNAMICS_MODEL}"
    ax.set_title(title + ")")
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_simulation_results(
    obstacle_history: List[List[Tuple[float, float, float]]],
    rrt_path: Optional[List[Tuple[float, float]]],
    executed_trajectory: np.ndarray,
    goal_position: Tuple[float, float],
    start_state: jnp.ndarray,
) -> None:
    """Replay with per-step static + dynamic obstacles (GIF export)."""
    title = f"RRT + SQP-MPC (plant={DYNAMICS_MODEL}"
    if MPC_DYNAMICS_MODEL != DYNAMICS_MODEL:
        title += f", mpc={MPC_DYNAMICS_MODEL}"
    if USE_DYNAMIC_OBSTACLES and DYNAMIC_OBSTACLE_FIELD is not None:
        title += f", dynamic={len(DYNAMIC_OBSTACLE_FIELD.movers)}"
    title += ")"

    def dynamic_disks_at_step(step: int) -> List[Tuple[float, float, float]]:
        if DYNAMIC_OBSTACLE_FIELD is None:
            return []
        return DYNAMIC_OBSTACLE_FIELD.disks_at_step(step, DT)

    if ANIMATION_SAVE_PATH:
        os.makedirs(os.path.dirname(ANIMATION_SAVE_PATH) or ".", exist_ok=True)
    animate_mpc_simulation(
        executed_trajectory,
        static_obstacle_list=OBSTACLE_LIST,
        obstacle_list=OBSTACLE_LIST if not USE_DYNAMIC_OBSTACLES else (),
        obstacles_at_step=dynamic_disks_at_step if USE_DYNAMIC_OBSTACLES else None,
        rrt_path=rrt_path,
        goal_position=goal_position,
        start_state=np.asarray(start_state),
        title=title,
        config=AnimationConfig(dt=DT, interval_ms=ANIMATION_INTERVAL_MS),
        save_path=ANIMATION_SAVE_PATH,
        show=False,
    )
    if ANIMATION_SAVE_PATH:
        print(f"Animation saved to {ANIMATION_SAVE_PATH}")


def main() -> None:
    """Run closed-loop RRT + MPC simulation and visualize results."""
    plant = get_dynamics(DYNAMICS_MODEL)
    mpc_model = get_dynamics(MPC_DYNAMICS_MODEL)
    print(
        f"[Dynamics] plant={DYNAMICS_MODEL}, "
        f"mpc_predictor={MPC_DYNAMICS_MODEL}"
    )
    print(f"[Scenario] start={BEAMNG_START_XY}, goal={BEAMNG_GOAL_XY}")
    print(
        f"[State clamps] x=[{BEAMNG_STATE_LOW[0]:.0f}, {BEAMNG_STATE_HIGH[0]:.0f}], "
        f"y=[{BEAMNG_STATE_LOW[1]:.0f}, {BEAMNG_STATE_HIGH[1]:.0f}]"
    )
    print(f"[Obstacles] static={len(OBSTACLE_LIST)}", end="")
    if DYNAMIC_OBSTACLE_FIELD is not None:
        print(f", dynamic={len(DYNAMIC_OBSTACLE_FIELD.movers)}", end="")
    print()

    # --- High-level planner (run once) ---
    rrt_path, _rrt_planner = run_rrt_planner()
    if rrt_path is None:
        print("RRT failed to find a path. Exiting.")
        return

    print(f"RRT path found with {len(rrt_path)} waypoints.")

    rrt_path = densify_polyline(
        smooth_path_moving_average(rrt_path, passes=2),
        spacing_m=PATH_SPACING_M,
    )
    print(f"Reference path: {len(rrt_path)} waypoints (~{PATH_SPACING_M} m spacing)")

    # --- Low-level controller ---
    use_fd_linearization = False
    mpc = build_mpc_controller(mpc_model, START_STATE)
    mpc.reset_path_tracking()

    # --- Simulation state and logs ---
    state = START_STATE
    state_history: List[jnp.ndarray] = [state]
    control_history: List[jnp.ndarray] = []
    obstacle_history: List[List[Tuple[float, float, float]]] = [
        world_obstacles_at_step(0)
    ]

    print(
        f"MPC: H={MPC_HORIZON}, sqp_iter={SQP_MAX_ITER}, "
        f"obstacles_in_qp={mpc.n_obs}, "
        f"jax_linearization={not use_fd_linearization}"
    )
    print("Running closed-loop simulation...")
    mpc_times: list[float] = []
    for step in range(MAX_SIM_STEPS):
        if step % 10 == 0:
            print(f"  step {step}/{MAX_SIM_STEPS}", flush=True)

        if mpc.n_obs > 0:
            mpc.update_obstacle_list(mpc_dynamic_obstacles_at_step(step))
            mpc.set_obstacle_horizon(mpc_obstacle_horizon(step))

        t0 = time.perf_counter()
        u = mpc.compute_action(state, rrt_path, wheel_base=WHEEL_BASE)
        mpc_times.append(time.perf_counter() - t0)
        control_history.append(u)
        state = plant.rk4_step(state, u, WHEEL_BASE, DT)
        state_history.append(state)
        obstacle_history.append(world_obstacles_at_step(step + 1))

        goal_xy = jnp.array(GOAL_POSITION)
        pos_err = float(jnp.linalg.norm(state[:2] - goal_xy))
        if pos_err < GOAL_THRESHOLD:
            print(f"  goal reached at step {step} (error={pos_err:.4f} m)")
            break

        # Stop if we overshot the goal while still moving quickly
        if pos_err < GOAL_SLOW_RADIUS and float(state[3]) < 0.15:
            print(f"  goal captured at step {step} (error={pos_err:.4f} m, v≈0)")
            break

    executed_trajectory = np.array(state_history)

    n_steps = len(state_history) - 1
    print(f"Simulation finished after {n_steps} steps.")
    if PROFILE_MPC and mpc_times:
        arr = np.asarray(mpc_times)
        print(
            f"MPC timing: mean {arr.mean():.3f}s/step, "
            f"max {arr.max():.3f}s, total {arr.sum():.1f}s"
        )
    final_dynamic = (
        DYNAMIC_OBSTACLE_FIELD.disks_at_step(len(obstacle_history) - 1, DT)
        if DYNAMIC_OBSTACLE_FIELD is not None
        else None
    )
    if SHOW_STATIC_PLOT:
        plot_simulation_results(
            static_obstacles=OBSTACLE_LIST,
            rrt_path=rrt_path,
            executed_trajectory=executed_trajectory,
            goal_position=GOAL_POSITION,
            start_state=START_STATE,
            dynamic_obstacles=final_dynamic,
        )
    if SHOW_ANIMATION:
        animate_simulation_results(
            obstacle_history=obstacle_history,
            rrt_path=rrt_path,
            executed_trajectory=executed_trajectory,
            goal_position=GOAL_POSITION,
            start_state=START_STATE,
        )


if __name__ == "__main__":
    main()