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
    BEAMNG_START_XY,
    BEAMNG_TARGET_SPEED_MPS,
    BEAMNG_X_BOUNDS,
    BEAMNG_Y_BOUNDS,
)
from sindympc.controllers.sqp_mpc import CONTROL_DIM, STATE_DIM, SQPMPC
from sindympc.dynamics import DynamicsName, get_dynamics
from sindympc.planners.rrt import RRT
from sindympc.utils.obstacles import (
    DEFAULT_MIN_DIST_FROM_POINTS,
    DEFAULT_NUM_OBSTACLES,
    DEFAULT_RADIUS_RANGE,
    sample_random_obstacles,
)
from sindympc.utils.path_utils import densify_polyline, smooth_path_moving_average

# -----------------------------------------------------------------------------
# Dynamics plant: "bicycle" (kinematic) or "sindy" (data/sindy_model.npz)
# RRT is geometric only; this switch affects MPC + closed-loop simulation.
# -----------------------------------------------------------------------------
DYNAMICS_MODEL: DynamicsName = "bicycle"
# DYNAMICS_MODEL: DynamicsName = "sindy"
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
    BEAMNG_X_BOUNDS,
    BEAMNG_Y_BOUNDS,
    DEFAULT_RADIUS_RANGE,
    RRT_START,
    RRT_GOAL,
    min_dist_from_points=DEFAULT_MIN_DIST_FROM_POINTS,
)

# RRT planner parameters (scaled for ~100+ m routes)
RRT_STEP_SIZE: float = BEAMNG_RRT_STEP_SIZE
RRT_MAX_ITER: int = 10000
PADDING_FRACTION: float = 0.05  # bounds already set explicitly

# Simulation parameters (dt matches telemetry used for SINDy)
DT: float = BEAMNG_DT
MAX_SIM_STEPS: int = 500
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
# Set SINDYMPC_MPC_OBSTACLES=1 to experiment with in-QP obstacles (not recommended).
USE_MPC_OBSTACLE_CONSTRAINTS: bool = os.environ.get(
    "SINDYMPC_MPC_OBSTACLES", ""
).lower() in ("1", "true", "yes")
PROFILE_MPC: bool = os.environ.get("SINDYMPC_PROFILE_MPC", "").lower() in (
    "1",
    "true",
    "yes",
)
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
SQP_MAX_ITER = 3
QN = jnp.diag(jnp.array([
    80.0,  # terminal x — pull onto goal at end of path
    80.0,  # terminal y
    8.0,
    1.0,
]))
OBSTACLE_MARGIN: float = 0.35
GOAL_SLOW_RADIUS: float = BEAMNG_GOAL_SLOW_RADIUS_M

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


def build_mpc_controller(mpc_dynamics: ModuleType) -> SQPMPC:
    """Construct SQP-MPC; ``mpc_dynamics`` is the model used inside the optimizer."""
    use_fd = False
    mpc_obstacles = OBSTACLE_LIST if USE_MPC_OBSTACLE_CONSTRAINTS else []
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
    obstacle_list: List[Tuple[float, float, float]],
    rrt_path: Optional[List[Tuple[float, float]]],
    executed_trajectory: np.ndarray,
    goal_position: Tuple[float, float],
    start_state: jnp.ndarray,
) -> None:
    """Plot obstacles, static RRT path, and MPC-executed trajectory.

    Args:
        obstacle_list: Circular obstacles as (x, y, radius).
        rrt_path: Global RRT waypoint path, or None if planning failed.
        executed_trajectory: Logged states, shape (N, STATE_DIM).
        goal_position: Target (x, y).
        start_state: Initial [x, y, theta, v].
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for ox, oy, radius in obstacle_list:
        ax.add_patch(Circle((ox, oy), radius, color="dimgray", alpha=0.6))

    if rrt_path is not None and len(rrt_path) > 0:
        rrt_xs, rrt_ys = zip(*rrt_path)
        ax.plot(rrt_xs, rrt_ys, "g--", linewidth=2, label="RRT path")

    if executed_trajectory.size > 0:
        ax.plot(
            executed_trajectory[:, 0],
            executed_trajectory[:, 1],
            "b-",
            linewidth=2,
            label="MPC trajectory",
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

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = f"RRT + SQP-MPC (plant={DYNAMICS_MODEL}"
    if MPC_DYNAMICS_MODEL != DYNAMICS_MODEL:
        title += f", mpc={MPC_DYNAMICS_MODEL}"
    ax.set_title(title + ")")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run closed-loop RRT + MPC simulation and visualize results."""
    plant = get_dynamics(DYNAMICS_MODEL)
    mpc_model = get_dynamics(MPC_DYNAMICS_MODEL)
    print(
        f"[Dynamics] plant={DYNAMICS_MODEL}, "
        f"mpc_predictor={MPC_DYNAMICS_MODEL}"
    )

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
    mpc_obstacles = OBSTACLE_LIST if USE_MPC_OBSTACLE_CONSTRAINTS else []
    mpc = build_mpc_controller(mpc_model)
    mpc.reset_path_tracking()

    # --- Simulation state and logs ---
    state = START_STATE
    state_history: List[jnp.ndarray] = [state]
    control_history: List[jnp.ndarray] = []

    print(
        f"MPC: H={MPC_HORIZON}, sqp_iter={SQP_MAX_ITER}, "
        f"obstacles_in_qp={len(mpc_obstacles)}, "
        f"jax_linearization={not use_fd_linearization}"
    )
    print("Running closed-loop simulation...")
    mpc_times: list[float] = []
    for step in range(MAX_SIM_STEPS):
        if step % 10 == 0:
            print(f"  step {step}/{MAX_SIM_STEPS}", flush=True)

        t0 = time.perf_counter()
        u = mpc.compute_action(state, rrt_path, wheel_base=WHEEL_BASE)
        mpc_times.append(time.perf_counter() - t0)
        control_history.append(u)
        state = plant.rk4_step(state, u, WHEEL_BASE, DT)
        state_history.append(state)

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
    plot_simulation_results(
        obstacle_list=OBSTACLE_LIST,
        rrt_path=rrt_path,
        executed_trajectory=executed_trajectory,
        goal_position=GOAL_POSITION,
        start_state=START_STATE,
    )


if __name__ == "__main__":
    main()