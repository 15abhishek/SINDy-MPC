"""Closed-loop simulation: RRT global plan + SQP-MPC tracking.

Pipeline:
    1. Plan once with RRT to obtain waypoint path.
    2. Simulate bicycle dynamics with receding-horizon SQP-MPC.
    3. Plot obstacles, RRT baseline path, and executed trajectory.
"""

from __future__ import annotations

import os

# Force CPU JAX if CUDA plugins are installed but the driver is incompatible.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from sindympc.controllers.sqp_mpc import CONTROL_DIM, STATE_DIM, SQPMPC
from sindympc.dynamics.bicycle_model import rk4_step
from sindympc.planners.rrt import RRT

# -----------------------------------------------------------------------------
# Map / scenario (aligned with sindympc/utils/visualize_rrt.py obstacles)
# -----------------------------------------------------------------------------

GOAL_POSITION: Tuple[float, float] = (10, 10)

_INIT_HEADING = float(np.arctan2(GOAL_POSITION[1] - 0.1, GOAL_POSITION[0] - 0.1))
START_STATE = jnp.array([0.1, 0.1, _INIT_HEADING, 0.5])  # [x, y, theta, v]

# RRT uses (x, y) start/goal; match the simulation start/goal positions
RRT_START: Tuple[float, float] = (float(START_STATE[0]), float(START_STATE[1]))
RRT_GOAL: Tuple[float, float] = GOAL_POSITION

OBSTACLE_LIST: List[Tuple[float, float, float]] = [
    # (x, y, radius)
    (5, 5, 1),
    # (0.2, 0.2, 0.05)
]

# RRT planner parameters (from visualize_rrt.py)
RRT_STEP_SIZE: float = 0.1
RRT_MAX_ITER: int = 5000

# Simulation parameters
DT: float = 0.05
MAX_SIM_STEPS: int = 500
GOAL_THRESHOLD: float = 0.5  # [m] position error to stop simulation

# Vehicle / MPC parameters
WHEEL_BASE: float = 2.5
MAX_ACCELERATION: float = 0.5
MAX_STEERING_ANGLE: float = 0.5
MPC_HORIZON: int = 25
TARGET_SPEED: float = 1.5

# Cost matrices (tune as needed)
Q = jnp.diag(jnp.array([
    10.0,  # x
    10.0,  # y
    5.0,   # theta
    1.0,   # velocity
]))
R = jnp.diag(jnp.array([
    0.5,   # acceleration penalty
    0.1,   # steering penalty (was too high — prevented turning)
]))
QN = jnp.diag(jnp.array([
    100.0,  # terminal x — pull onto goal at end of path
    100.0,  # terminal y
    5.0,
    1.0,
]))
OBSTACLE_MARGIN: float = 0.35
GOAL_SLOW_RADIUS: float = 3.0  # mirrors controller goal-phase logic

def run_rrt_planner() -> Tuple[Optional[List[Tuple[float, float]]], RRT]:
    """Run RRT once and return waypoints plus the planner instance."""
    planner = RRT(
        start=RRT_START,
        goal=RRT_GOAL,
        obstacle_list=OBSTACLE_LIST,
        step_size=RRT_STEP_SIZE,
        max_iter=RRT_MAX_ITER,
    )
    path = planner.plan()
    return path, planner


def build_mpc_controller() -> SQPMPC:
    """Construct the SQP-MPC controller with simulation tuning parameters."""
    return SQPMPC(
        Q=Q,
        R=R,
        H=MPC_HORIZON,
        dt=DT,
        max_acceleration=MAX_ACCELERATION,
        max_steering_angle=MAX_STEERING_ANGLE,
        wheel_base=WHEEL_BASE,
        sqp_max_iter=3,
        solver="OSQP",
        target_speed=TARGET_SPEED,
        QN=QN,
        obstacle_list=OBSTACLE_LIST,
        obstacle_margin=OBSTACLE_MARGIN,
        goal_position=GOAL_POSITION,
    )


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
    ax.set_title("RRT + SQP-MPC Closed-Loop Simulation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run closed-loop RRT + MPC simulation and visualize results."""
    # --- High-level planner (run once) ---
    rrt_path, _rrt_planner = run_rrt_planner()
    if rrt_path is None:
        print("RRT failed to find a path. Exiting.")
        return

    print(f"RRT path found with {len(rrt_path)} waypoints.")

    # --- Low-level controller ---
    mpc = build_mpc_controller()
    mpc.reset_path_tracking()

    # --- Simulation state and logs ---
    state = START_STATE
    state_history: List[jnp.ndarray] = [state]
    control_history: List[jnp.ndarray] = []

    print("Running closed-loop simulation...")
    for step in range(MAX_SIM_STEPS):
        if step % 10 == 0:
            print(f"  step {step}/{MAX_SIM_STEPS}", flush=True)

        u = mpc.compute_action(state, rrt_path, wheel_base=WHEEL_BASE)
        control_history.append(u)
        state = rk4_step(state, u, WHEEL_BASE, DT)
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

    print(f"Simulation finished after {len(state_history) - 1} steps.")
    plot_simulation_results(
        obstacle_list=OBSTACLE_LIST,
        rrt_path=rrt_path,
        executed_trajectory=executed_trajectory,
        goal_position=GOAL_POSITION,
        start_state=START_STATE,
    )


if __name__ == "__main__":
    main()