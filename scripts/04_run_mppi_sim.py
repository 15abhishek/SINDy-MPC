"""Closed-loop simulation: RRT global plan + MPPI tracking.

Same scenario and plant options as ``03_run_mpc_sim.py``, but the local
controller is :class:`~sindympc.controllers.mppi.MPPIController` instead of SQP-MPC.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import List, Optional, Tuple

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
    BEAMNG_SCENARIO_X_BOUNDS,
    BEAMNG_SCENARIO_Y_BOUNDS,
)
from sindympc.controllers.mppi import MPPIController
from sindympc.controllers.path_reference import path_tracking_stats
from sindympc.dynamics import DynamicsName, get_dynamics
from sindympc.planners.rrt import RRT
from sindympc.utils.obstacles import get_scenario_obstacles
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
# Dynamics plant: "bicycle" or "sindy"
# -----------------------------------------------------------------------------
# DYNAMICS_MODEL: DynamicsName = "sindy"
DYNAMICS_MODEL: DynamicsName = "bicycle"
_plant_override = os.environ.get("SINDYMPC_PLANT", "").strip().lower()
if _plant_override in ("bicycle", "sindy"):
    DYNAMICS_MODEL = _plant_override  # type: ignore[assignment]
CONTROLLER_DYNAMICS_MODEL: DynamicsName = DYNAMICS_MODEL
_USE_SINDY_PLANT = DYNAMICS_MODEL == "sindy"

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
)

RRT_START: Tuple[float, float] = BEAMNG_START_XY
RRT_GOAL: Tuple[float, float] = GOAL_POSITION
# Sample RRT in the start–goal corridor (not the full ±500 m BeamNG map).
RRT_BOUNDS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    BEAMNG_SCENARIO_X_BOUNDS,
    BEAMNG_SCENARIO_Y_BOUNDS,
)

SIM_SEED: int = int(os.environ.get("SINDYMPC_SEED", "42"))

OBSTACLE_LIST: List[Tuple[float, float, float]] = get_scenario_obstacles()

RRT_STEP_SIZE: float = BEAMNG_RRT_STEP_SIZE
RRT_MAX_ITER: int = 10000
PADDING_FRACTION: float = 0.05

DT: float = BEAMNG_DT
_MAX_ROUTE_M = float(
    np.hypot(
        BEAMNG_GOAL_XY[0] - BEAMNG_START_XY[0],
        BEAMNG_GOAL_XY[1] - BEAMNG_START_XY[1],
    )
)
MAX_SIM_STEPS: int = int(np.ceil(_MAX_ROUTE_M / (BEAMNG_TARGET_SPEED_MPS * DT))) + 150
GOAL_THRESHOLD: float = BEAMNG_GOAL_THRESHOLD_M
GOAL_SLOW_RADIUS: float = BEAMNG_GOAL_SLOW_RADIUS_M

WHEEL_BASE: float = 2.5
MAX_ACCELERATION: float = 1.0
MAX_STEERING_ANGLE: float = 1.0
HORIZON: int = int(os.environ.get("SINDYMPC_MPPI_HORIZON", "25"))
PATH_SPACING_M: float = 3.0
# Cruise reference for MPPI (lower than BeamNG max — easier for SINDy to brake at goal).
TARGET_SPEED: float = float(
    os.environ.get(
        "SINDYMPC_MPPI_TARGET_SPEED",
        "3.5" if _USE_SINDY_PLANT else str(BEAMNG_TARGET_SPEED_MPS),
    )
)

# MPPI hyperparameters (HybridVLA-style sampling + softmax)
MPPI_NUM_SAMPLES: int = int(os.environ.get("SINDYMPC_MPPI_SAMPLES", "256"))
MPPI_LAMBDA: float = float(os.environ.get("SINDYMPC_MPPI_LAMBDA", "1.0"))
MPPI_NOISE_ACCEL: float = float(os.environ.get("SINDYMPC_MPPI_NOISE_ACCEL", "0.35"))
MPPI_NOISE_STEER: float = float(os.environ.get("SINDYMPC_MPPI_NOISE_STEER", "0.12"))
MPPI_SEED: int = int(os.environ.get("SINDYMPC_MPPI_SEED", str(SIM_SEED)))
MPPI_W_CROSS_TRACK: float = float(os.environ.get("SINDYMPC_MPPI_W_PATH", "30.0"))

# Optional obstacle penalty in rollouts (RRT still plans around static disks)
USE_MPPI_OBSTACLE_COST: bool = os.environ.get(
    "SINDYMPC_MPPI_OBSTACLES", ""
).lower() in ("1", "true", "yes")

PROFILE_CONTROLLER: bool = os.environ.get("SINDYMPC_PROFILE_MPC", "").lower() in (
    "1",
    "true",
    "yes",
)
SHOW_STATIC_PLOT: bool = os.environ.get("SINDYMPC_STATIC_PLOT", "1").lower() in (
    "1",
    "true",
    "yes",
)
SHOW_ANIMATION: bool = os.environ.get("SINDYMPC_ANIMATE", "0").lower() in (
    "1",
    "true",
    "yes",
)
_DEFAULT_GIF = os.path.join("results", "mppi_replay.gif")
ANIMATION_SAVE_PATH: Optional[str] = (
    os.environ.get("SINDYMPC_ANIMATE_SAVE")
    or (_DEFAULT_GIF if SHOW_ANIMATION else None)
)
ANIMATION_INTERVAL_MS: int = int(os.environ.get("SINDYMPC_ANIMATE_INTERVAL_MS", "50"))

Q = jnp.diag(jnp.array([15.0, 15.0, 3.0, 2.0 if _USE_SINDY_PLANT else 0.1]))
R_ACCEL = 0.05 if _USE_SINDY_PLANT else 0.03
R_STEER = 0.2 if _USE_SINDY_PLANT else 0.005
R = jnp.diag(jnp.array([R_ACCEL, R_STEER]))
R_DELTA = jnp.diag(
    jnp.array([0.08, 12.0] if _USE_SINDY_PLANT else [0.1, 1.0])
)
QN = jnp.diag(jnp.array([80.0, 80.0, 8.0, 5.0 if _USE_SINDY_PLANT else 1.0]))
MPPI_W_GOAL_XY: float = float(os.environ.get("SINDYMPC_MPPI_W_GOAL", "0"))
MPPI_W_GOAL_V: float = float(os.environ.get("SINDYMPC_MPPI_W_GOAL_V", "0"))
OBSTACLE_MARGIN: float = 1.0


def run_rrt_planner() -> Tuple[Optional[List[Tuple[float, float]]], RRT]:
    planner = RRT(
        start=RRT_START,
        goal=RRT_GOAL,
        bounds=RRT_BOUNDS,
        padding_fraction=PADDING_FRACTION,
        obstacle_list=OBSTACLE_LIST,
        step_size=RRT_STEP_SIZE,
        max_iter=RRT_MAX_ITER,
        seed=SIM_SEED,
    )
    return planner.plan(), planner


def build_mppi_controller(controller_dynamics: ModuleType) -> MPPIController:
    obstacles = list(OBSTACLE_LIST) if USE_MPPI_OBSTACLE_COST else []
    return MPPIController(
        Q=Q,
        R=R,
        QN=QN,
        H=HORIZON,
        dt=DT,
        max_acceleration=MAX_ACCELERATION,
        max_steering_angle=MAX_STEERING_ANGLE,
        num_samples=MPPI_NUM_SAMPLES,
        lambda_=MPPI_LAMBDA,
        noise_accel=MPPI_NOISE_ACCEL,
        noise_steer=MPPI_NOISE_STEER,
        target_speed=TARGET_SPEED,
        wheel_base=WHEEL_BASE,
        goal_position=GOAL_POSITION,
        rk4_step_fn=controller_dynamics.rk4_step,
        obstacle_list=obstacles,
        obstacle_margin=OBSTACLE_MARGIN,
        R_delta=R_DELTA,
        w_cross_track=MPPI_W_CROSS_TRACK,
        w_goal_xy=MPPI_W_GOAL_XY,
        w_goal_v=MPPI_W_GOAL_V,
        goal_slow_radius=GOAL_SLOW_RADIUS,
        seed=MPPI_SEED,
    )


def plot_simulation_results(
    obstacle_list: List[Tuple[float, float, float]],
    rrt_path: Optional[List[Tuple[float, float]]],
    executed_trajectory: np.ndarray,
    goal_position: Tuple[float, float],
    start_state: jnp.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for ox, oy, radius in obstacle_list:
        ax.add_patch(Circle((ox, oy), radius, color="dimgray", alpha=0.6))
    if rrt_path:
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
    ax.plot(float(start_state[0]), float(start_state[1]), "go", markersize=10, label="Start")
    ax.plot(goal_position[0], goal_position[1], "r*", markersize=14, label="Goal")
    extra_xy = [goal_position, (float(start_state[0]), float(start_state[1]))]
    if rrt_path:
        extra_xy.extend(rrt_path)
    apply_scenario_axes(
        ax,
        trajectory_xy=executed_trajectory[:, :2] if executed_trajectory.size else None,
        extra_xy=extra_xy,
    )
    title = f"RRT + MPPI (plant={DYNAMICS_MODEL})"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_simulation_results(
    rrt_path: Optional[List[Tuple[float, float]]],
    executed_trajectory: np.ndarray,
    goal_position: Tuple[float, float],
    start_state: jnp.ndarray,
) -> None:
    if ANIMATION_SAVE_PATH:
        os.makedirs(os.path.dirname(ANIMATION_SAVE_PATH) or ".", exist_ok=True)
    animate_mpc_simulation(
        executed_trajectory,
        obstacle_list=OBSTACLE_LIST,
        rrt_path=rrt_path,
        goal_position=goal_position,
        start_state=np.asarray(start_state),
        title=f"RRT + MPPI (plant={DYNAMICS_MODEL})",
        config=AnimationConfig(dt=DT, interval_ms=ANIMATION_INTERVAL_MS),
        save_path=ANIMATION_SAVE_PATH,
        show=False,
    )
    if ANIMATION_SAVE_PATH:
        print(f"Animation saved to {ANIMATION_SAVE_PATH}")


def main() -> None:
    plant = get_dynamics(DYNAMICS_MODEL)
    ctrl_model = get_dynamics(CONTROLLER_DYNAMICS_MODEL)
    print(
        f"[Dynamics] plant={DYNAMICS_MODEL}, "
        f"controller_model={CONTROLLER_DYNAMICS_MODEL}"
    )
    print(f"[Scenario] start={BEAMNG_START_XY}, goal={BEAMNG_GOAL_XY}")
    print(
        f"[State clamps] x=[{BEAMNG_STATE_LOW[0]:.0f}, {BEAMNG_STATE_HIGH[0]:.0f}], "
        f"y=[{BEAMNG_STATE_LOW[1]:.0f}, {BEAMNG_STATE_HIGH[1]:.0f}]"
    )
    print(
        f"[Obstacles] static={len(OBSTACLE_LIST)} (RRT); "
        f"mppi_obstacle_cost={USE_MPPI_OBSTACLE_COST}"
    )

    rrt_path, _ = run_rrt_planner()
    if rrt_path is None:
        print("RRT failed to find a path. Exiting.")
        return

    print(f"RRT path found with {len(rrt_path)} waypoints.")
    rrt_path = densify_polyline(
        smooth_path_moving_average(rrt_path, passes=2),
        spacing_m=PATH_SPACING_M,
    )
    print(f"Reference path: {len(rrt_path)} waypoints (~{PATH_SPACING_M} m spacing)")

    controller = build_mppi_controller(ctrl_model)
    controller.reset_path_tracking()

    print(
        f"MPPI: H={HORIZON}, K={MPPI_NUM_SAMPLES}, lambda={MPPI_LAMBDA}, "
        f"obstacles_in_cost={len(controller.obstacle_list)}"
    )
    print("Running closed-loop simulation...")

    state = START_STATE
    state_history: List[jnp.ndarray] = [state]
    ctrl_times: list[float] = []

    for step in range(MAX_SIM_STEPS):
        if step % 10 == 0:
            print(f"  step {step}/{MAX_SIM_STEPS}", flush=True)

        t0 = time.perf_counter()
        u = controller.compute_action(state, rrt_path, wheel_base=WHEEL_BASE)
        ctrl_times.append(time.perf_counter() - t0)
        state = plant.rk4_step(state, u, WHEEL_BASE, DT)
        state_history.append(state)

        pos_err = float(jnp.linalg.norm(state[:2] - jnp.array(GOAL_POSITION)))
        if pos_err < GOAL_THRESHOLD:
            print(f"  goal reached at step {step} (error={pos_err:.4f} m)")
            break
        if pos_err < GOAL_SLOW_RADIUS and float(state[3]) < 0.15:
            print(f"  goal captured at step {step} (error={pos_err:.4f} m, v≈0)")
            break

    executed_trajectory = np.array(state_history)
    n_steps = len(state_history) - 1
    final_err = float(jnp.linalg.norm(state[:2] - jnp.array(GOAL_POSITION)))
    print(f"Simulation finished after {n_steps} steps (goal error={final_err:.2f} m).")
    if final_err >= GOAL_THRESHOLD:
        print("  Did not reach goal within MAX_SIM_STEPS.")
    stats = path_tracking_stats(
        executed_trajectory[:, :2],
        rrt_path,
        controller.last_arc_s,
    )
    print(
        f"[Path tracking] progress={stats['path_progress_pct']:.1f}% of RRT length, "
        f"mean cross-track={stats['mean_cross_track_m']:.2f} m, "
        f"max={stats['max_cross_track_m']:.2f} m"
    )
    if PROFILE_CONTROLLER and ctrl_times:
        arr = np.asarray(ctrl_times)
        print(
            f"MPPI timing: mean {arr.mean():.3f}s/step, "
            f"max {arr.max():.3f}s, total {arr.sum():.1f}s"
        )

    if SHOW_STATIC_PLOT:
        plot_simulation_results(
            obstacle_list=OBSTACLE_LIST,
            rrt_path=rrt_path,
            executed_trajectory=executed_trajectory,
            goal_position=GOAL_POSITION,
            start_state=START_STATE,
        )
    if SHOW_ANIMATION:
        animate_simulation_results(
            rrt_path=rrt_path,
            executed_trajectory=executed_trajectory,
            goal_position=GOAL_POSITION,
            start_state=START_STATE,
        )


if __name__ == "__main__":
    main()
