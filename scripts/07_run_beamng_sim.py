"""Live BeamNG closed loop: RRT global plan + SQP-MPC or MPPI tracking.

Connects to BeamNG.tech using the lab socket configuration (``localhost:25252``),
plans once with RRT in the SINDy-MPC scenario frame, then runs a real-time control
loop that polls vehicle telemetry, solves the tracking problem, and sends actuator
commands back through ``vehicle.control(...)``.

Usage
-----
# SQP-MPC (default), SINDy dynamics
python scripts/07_run_beamng_sim.py

# MPPI + kinematic bicycle model
python scripts/07_run_beamng_sim.py --controller mppi --dynamics bicycle

# Attach to running BeamNG, save plot without displaying
python scripts/07_run_beamng_sim.py --controller mpc --dynamics sindy --no-launch --show-plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from matplotlib.patches import Circle

from types import ModuleType

from sindympc.beamng.helpers import (
    BeamNGActuatorMapper,
    poll_vehicle_state_4d,
    quat_from_heading,
)
from sindympc.config.scenario_obstacles import BEAMNG_OBSTACLE_GROUND_Z
from sindympc.config.beamng_defaults import (
    BEAMNG_DT,
    BEAMNG_GOAL_FULL_SPEED_RADIUS_M,
    BEAMNG_GOAL_SLOW_RADIUS_M,
    BEAMNG_GOAL_THRESHOLD_M,
    BEAMNG_GOAL_XY,
    BEAMNG_RRT_STEP_SIZE,
    BEAMNG_SCENARIO_X_BOUNDS,
    BEAMNG_SCENARIO_Y_BOUNDS,
    BEAMNG_SPAWN_POS,
    BEAMNG_SPAWN_ROT,
    BEAMNG_START_XY,
    BEAMNG_TARGET_SPEED_MPS,
)
from sindympc.beamng.scenario_obstacles import add_obstacle_cylinders
from sindympc.controllers.mppi import MPPIController
from sindympc.controllers.sqp_mpc import SQPMPC
from sindympc.dynamics import DynamicsName, get_dynamics
from sindympc.planners.rrt import RRT
from sindympc.utils.obstacles import get_scenario_obstacles
from sindympc.utils.path_utils import chaikin_smooth, densify_polyline, smooth_path_moving_average
from sindympc.utils.plot_utils import apply_scenario_axes
from sindympc.utils.benchmark_metrics import compute_max_steps, summarize_run
from sindympc.utils.sim_animation import TRAIL_COLOR, TRAIL_LABEL, TRAIL_LINEWIDTH

# -----------------------------------------------------------------------------
# Lab BeamNG connection defaults (HybridVLA hybridvla/beamng/config.py)
# -----------------------------------------------------------------------------
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 25252
DEFAULT_BNG_HOME = os.path.expanduser("~/BeamNG.tech.v0.38.5.0")
DEFAULT_MAP = "gridmap_v2"
DEFAULT_VEHICLE_MODEL = "utv"
DEFAULT_VEHICLE_ID = "ego_vehicle"
DEFAULT_SCENARIO_NAME = "sindympc_live_tracking"

SIM_HZ = 60
STEPS_PER_CTRL = max(1, int(round(BEAMNG_DT * SIM_HZ)))

GOAL_POSITION: Tuple[float, float] = BEAMNG_GOAL_XY
RRT_START: Tuple[float, float] = BEAMNG_START_XY
RRT_GOAL: Tuple[float, float] = BEAMNG_GOAL_XY
RRT_BOUNDS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    BEAMNG_SCENARIO_X_BOUNDS,
    BEAMNG_SCENARIO_Y_BOUNDS,
)

SIM_SEED = int(os.environ.get("SINDYMPC_SEED", "42"))
USE_SCENARIO_OBSTACLES = os.environ.get("SINDYMPC_OBSTACLES", "1").lower() not in (
    "0",
    "false",
    "no",
)
OBSTACLE_LIST: List[Tuple[float, float, float]] = (
    get_scenario_obstacles() if USE_SCENARIO_OBSTACLES else []
)

DT = BEAMNG_DT
WHEEL_BASE = 2.5
MAX_ACCELERATION = 1.0
MAX_STEERING_ANGLE = 1.0
MPC_HORIZON = 25
PATH_SPACING_M = 2.0
PATH_SMOOTH_PASSES = 3
PATH_CHAIKIN_ITERS = 2
TARGET_SPEED = BEAMNG_TARGET_SPEED_MPS
GOAL_THRESHOLD = BEAMNG_GOAL_THRESHOLD_M
GOAL_SLOW_RADIUS = BEAMNG_GOAL_SLOW_RADIUS_M
GOAL_FULL_SPEED_RADIUS = BEAMNG_GOAL_FULL_SPEED_RADIUS_M

_MAX_ROUTE_M = float(
    np.hypot(
        BEAMNG_GOAL_XY[0] - BEAMNG_START_XY[0],
        BEAMNG_GOAL_XY[1] - BEAMNG_START_XY[1],
    )
)
DEFAULT_MAX_STEPS = compute_max_steps(_MAX_ROUTE_M, TARGET_SPEED, DT)


def resolve_target_speed(
    args: argparse.Namespace,
    *,
    controller_key: str,
    use_sindy: bool,
) -> float:
    """CLI ``--target-speed`` overrides per-controller env defaults."""
    if getattr(args, "target_speed", None) is not None:
        return float(args.target_speed)
    if controller_key == "sqp":
        env_key = "SINDYMPC_SQP_TARGET_SPEED"
        default = str(TARGET_SPEED)
    else:
        env_key = "SINDYMPC_MPPI_TARGET_SPEED"
        default = str(TARGET_SPEED)
    return float(os.environ.get(env_key, default))

Controller = Union[SQPMPC, MPPIController]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RRT + MPC/MPPI live tracking in BeamNG.tech",
    )
    parser.add_argument("--host", default=os.environ.get("SINDYMPC_BNG_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SINDYMPC_BNG_PORT", str(DEFAULT_PORT))),
    )
    parser.add_argument(
        "--bng-home",
        default=os.environ.get("SINDYMPC_BNG_HOME", DEFAULT_BNG_HOME),
        help="BeamNG.tech install directory",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Connect to an already running BeamNG instance (launch=False)",
    )
    parser.add_argument("--map", default=os.environ.get("SINDYMPC_BNG_MAP", DEFAULT_MAP))
    parser.add_argument("--vehicle-model", default=DEFAULT_VEHICLE_MODEL)
    parser.add_argument(
        "--controller",
        choices=("sqp", "mpc", "mppi"),
        default=os.environ.get("SINDYMPC_CONTROLLER", "sqp").lower(),
        help="Tracking controller: sqp/mpc (SQP-MPC) or mppi",
    )
    parser.add_argument(
        "--dynamics",
        "--plant",
        dest="dynamics",
        choices=("bicycle", "sindy"),
        default=os.environ.get("SINDYMPC_PLANT", "sindy").lower(),
        help="Rollout dynamics inside the controller: bicycle or sindy",
    )
    parser.add_argument("--spawn-x", type=float, default=BEAMNG_SPAWN_POS[0])
    parser.add_argument("--spawn-y", type=float, default=BEAMNG_SPAWN_POS[1])
    parser.add_argument(
        "--spawn-z",
        type=float,
        default=BEAMNG_SPAWN_POS[2],
    )
    parser.add_argument(
        "--spawn-heading",
        type=float,
        default=None,
        help="Override spawn yaw [rad]; default aligns with first RRT segment",
    )
    parser.add_argument(
        "--use-config-rot",
        action="store_true",
        help="Use BEAMNG_SPAWN_ROT from config instead of path-aligned heading",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=None,
        metavar="MPS",
        help="Cruise speed setpoint for MPC/MPPI [m/s] (default: env or BEAMNG_TARGET_SPEED_MPS)",
    )
    parser.add_argument(
        "--actuator-profile",
        choices=("responsive", "smooth"),
        default=os.environ.get("SINDYMPC_BNG_ACTUATOR_PROFILE", "responsive"),
        help="BeamNG throttle/brake mapping: responsive (SQP) vs smooth (MPPI-like)",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Physics steps after scenario start before closed-loop control",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("results") / "beamng_live.jsonl",
        help="Optional JSONL log of state/control each tick (empty to disable)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=os.environ.get("SINDYMPC_PROFILE_MPC", "").lower()
        in ("1", "true", "yes"),
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Save end-of-run XY plot (default: results/beamng_<controller>_<dynamics>.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the end-of-run trajectory plot",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot interactively (always saved unless --no-plot)",
    )
    return parser.parse_args()


def normalize_controller_name(name: str) -> str:
    """Map CLI controller name to internal builder key."""
    if name in ("sqp", "mpc"):
        return "sqp"
    return "mppi"


def default_plot_path(controller: str, dynamics: str) -> Path:
    ctrl = normalize_controller_name(controller)
    return Path("results") / f"beamng_{ctrl}_{dynamics}.png"


def run_rrt_planner() -> Optional[List[Tuple[float, float]]]:
    planner = RRT(
        start=RRT_START,
        goal=RRT_GOAL,
        bounds=RRT_BOUNDS,
        padding_fraction=0.05,
        obstacle_list=OBSTACLE_LIST,
        step_size=BEAMNG_RRT_STEP_SIZE,
        max_iter=20000,
        seed=SIM_SEED,
    )
    return planner.plan()


def build_sqp_controller(
    mpc_dynamics: ModuleType,
    *,
    use_sindy: bool,
    target_speed: float,
) -> SQPMPC:
    use_obstacles = os.environ.get("SINDYMPC_MPC_OBSTACLES", "").lower() in (
        "1",
        "true",
        "yes",
    )
    # JAX sim uses R_steer≈0.005 for bicycle; BeamNG needs more steering penalty + Δu damping.
    # BeamNG SINDy SQP: strong v tracking + low accel penalty; v_ref uses full target
    # speed when far from goal (see sqp_mpc._waypoints_to_reference).
    if use_sindy:
        q = jnp.diag(jnp.array([15.0, 15.0, 3.0, 15.0]))
        qn = jnp.diag(jnp.array([80.0, 80.0, 8.0, 12.0]))
        r_steer = 0.12
        r_accel = 0.01
        r_delta = jnp.diag(jnp.array([0.05, 6.0]))
    else:
        q = jnp.diag(jnp.array([15.0, 15.0, 4.0, 0.1]))
        qn = jnp.diag(jnp.array([80.0, 80.0, 8.0, 1.0]))
        r_steer = 0.08
        r_accel = 0.03
        r_delta = jnp.diag(jnp.array([0.1, 8.0]))
    r = jnp.diag(jnp.array([r_accel, r_steer]))
    kwargs = dict(
        Q=q,
        R=r,
        H=MPC_HORIZON,
        dt=DT,
        max_acceleration=MAX_ACCELERATION,
        max_steering_angle=MAX_STEERING_ANGLE,
        wheel_base=WHEEL_BASE,
        sqp_max_iter=3,
        solver="OSQP",
        target_speed=target_speed,
        QN=qn,
        obstacle_list=list(OBSTACLE_LIST) if use_obstacles else [],
        obstacle_margin=1.0,
        goal_position=GOAL_POSITION,
        rk4_step_fn=mpc_dynamics.rk4_step,
        use_fd_linearization=False,
        R_delta=r_delta,
        verbose=False,
    )
    if use_sindy:
        kwargs["R_trp"] = jnp.diag(jnp.array([0.012, 0.03]))
    return SQPMPC(**kwargs)


def build_mppi_controller(
    controller_dynamics: ModuleType,
    *,
    use_sindy: bool,
    target_speed: float,
) -> MPPIController:
    use_obstacles = os.environ.get("SINDYMPC_MPPI_OBSTACLES", "").lower() in (
        "1",
        "true",
        "yes",
    )
    target = target_speed
    return MPPIController(
        Q=jnp.diag(jnp.array([15.0, 15.0, 3.0, 4.0 if use_sindy else 0.1])),
        R=jnp.diag(jnp.array([0.04 if use_sindy else 0.03, 0.18 if use_sindy else 0.005])),
        QN=jnp.diag(jnp.array([80.0, 80.0, 8.0, 6.0 if use_sindy else 1.0])),
        H=int(os.environ.get("SINDYMPC_MPPI_HORIZON", str(MPC_HORIZON))),
        dt=DT,
        max_acceleration=MAX_ACCELERATION,
        max_steering_angle=MAX_STEERING_ANGLE,
        num_samples=int(os.environ.get("SINDYMPC_MPPI_SAMPLES", "256")),
        lambda_=float(os.environ.get("SINDYMPC_MPPI_LAMBDA", "1.0")),
        noise_accel=float(os.environ.get("SINDYMPC_MPPI_NOISE_ACCEL", "0.35")),
        noise_steer=float(os.environ.get("SINDYMPC_MPPI_NOISE_STEER", "0.12")),
        target_speed=target,
        wheel_base=WHEEL_BASE,
        goal_position=GOAL_POSITION,
        rk4_step_fn=controller_dynamics.rk4_step,
        obstacle_list=list(OBSTACLE_LIST) if use_obstacles else [],
        obstacle_margin=1.0,
        R_delta=jnp.diag(jnp.array([0.08, 12.0] if use_sindy else [0.1, 1.0])),
        w_cross_track=float(os.environ.get("SINDYMPC_MPPI_W_PATH", "30.0")),
        goal_slow_radius=GOAL_SLOW_RADIUS,
        seed=int(os.environ.get("SINDYMPC_MPPI_SEED", str(SIM_SEED))),
    )


def build_controller(
    name: str,
    dynamics: ModuleType,
    *,
    use_sindy: bool,
    target_speed: float,
) -> Controller:
    if name == "mppi":
        return build_mppi_controller(
            dynamics, use_sindy=use_sindy, target_speed=target_speed,
        )
    return build_sqp_controller(
        dynamics, use_sindy=use_sindy, target_speed=target_speed,
    )


def path_start_heading(rrt_path: List[Tuple[float, float]]) -> float:
    """Yaw [rad] of the first RRT segment (fallback: start→goal)."""
    if len(rrt_path) >= 2:
        dx = rrt_path[1][0] - rrt_path[0][0]
        dy = rrt_path[1][1] - rrt_path[0][1]
        if abs(dx) + abs(dy) > 1e-6:
            return float(np.arctan2(dy, dx))
    return float(
        np.arctan2(
            BEAMNG_GOAL_XY[1] - BEAMNG_START_XY[1],
            BEAMNG_GOAL_XY[0] - BEAMNG_START_XY[0],
        )
    )


def resolve_spawn_rotation(
    args: argparse.Namespace,
    rrt_path: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """Spawn quaternion from path heading unless config rot is requested."""
    if args.use_config_rot:
        return BEAMNG_SPAWN_ROT
    heading = (
        args.spawn_heading
        if args.spawn_heading is not None
        else path_start_heading(rrt_path)
    )
    return quat_from_heading(heading)


def build_beamng_scenario(
    bng: BeamNGpy,
    args: argparse.Namespace,
    spawn_rot: Tuple[float, float, float, float],
) -> Tuple[Scenario, Vehicle]:
    scenario = Scenario(
        args.map,
        DEFAULT_SCENARIO_NAME,
        description="SINDy-MPC live RRT + tracking evaluation",
    )
    vehicle = Vehicle(
        DEFAULT_VEHICLE_ID,
        model=args.vehicle_model,
        license="SINDY",
        color="Red",
    )
    spawn_pos = (args.spawn_x, args.spawn_y, args.spawn_z)
    scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot)
    if OBSTACLE_LIST:
        ground_z = float(os.environ.get("SINDYMPC_OBSTACLE_GROUND_Z", BEAMNG_OBSTACLE_GROUND_Z))
        n_meshes = add_obstacle_cylinders(
            scenario,
            OBSTACLE_LIST,
            ground_z=ground_z,
        )
        print(
            f"[Sim] Spawned {n_meshes} BeamNG obstacle cylinder(s) "
            f"at z={ground_z:.2f} m"
        )
    # beamngpy>=1.28 attaches a "state" sensor in Vehicle.__init__
    return scenario, vehicle


def plot_beamng_run(
    *,
    executed_trajectory: np.ndarray,
    rrt_path: List[Tuple[float, float]],
    obstacle_list: List[Tuple[float, float, float]],
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    controller: str,
    dynamics: str,
    save_path: Path,
    show: bool,
) -> None:
    """Plot start, goal, RRT plan, obstacles, and executed BeamNG path."""
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

    ax.plot(start_xy[0], start_xy[1], "go", markersize=10, label="Start", zorder=5)
    ax.plot(goal_xy[0], goal_xy[1], "r*", markersize=14, label="Goal", zorder=5)

    extra_xy = [goal_xy, start_xy, *rrt_path]
    apply_scenario_axes(
        ax,
        trajectory_xy=executed_trajectory[:, :2] if executed_trajectory.size else None,
        extra_xy=extra_xy,
    )
    ctrl_label = "SQP-MPC" if normalize_controller_name(controller) == "sqp" else "MPPI"
    ax.set_title(f"BeamNG live run — RRT + {ctrl_label} (dynamics={dynamics})")
    ax.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[Plot] Saved trajectory plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def goal_reached(state: np.ndarray) -> bool:
    pos_err = float(np.hypot(state[0] - GOAL_POSITION[0], state[1] - GOAL_POSITION[1]))
    if pos_err < GOAL_THRESHOLD:
        return True
    if pos_err < GOAL_SLOW_RADIUS and state[3] < 0.15:
        return True
    return False


def run_live_loop(
    bng: BeamNGpy,
    vehicle: Vehicle,
    controller: Controller,
    rrt_path: List[Tuple[float, float]],
    args: argparse.Namespace,
) -> np.ndarray:
    ctrl_key = normalize_controller_name(args.controller)
    plant = "sindy" if args.dynamics == "sindy" else "bicycle"
    actuator_mapper = BeamNGActuatorMapper.for_controller(
        ctrl_key,
        dt=DT,
        max_acceleration=MAX_ACCELERATION,
        max_steering_angle=MAX_STEERING_ANGLE,
        plant=plant,  # type: ignore[arg-type]
        profile=getattr(args, "actuator_profile", "responsive"),  # type: ignore[arg-type]
        target_speed_mps=float(getattr(args, "target_speed_mps", TARGET_SPEED)),
        goal_position=GOAL_POSITION,
        goal_slow_radius_m=GOAL_SLOW_RADIUS,
        full_speed_goal_radius_m=GOAL_FULL_SPEED_RADIUS,
        sqp_cruise_governor=(ctrl_key == "sqp"),
    )
    if os.environ.get("SINDYMPC_BNG_SPEED_HORIZON"):
        actuator_mapper.speed_horizon_s = float(os.environ["SINDYMPC_BNG_SPEED_HORIZON"])
    if os.environ.get("SINDYMPC_BNG_SPEED_SMOOTH"):
        actuator_mapper.desired_speed_alpha = float(os.environ["SINDYMPC_BNG_SPEED_SMOOTH"])
    if os.environ.get("SINDYMPC_BNG_STEER_SLEW"):
        actuator_mapper.steer_max_delta = float(os.environ["SINDYMPC_BNG_STEER_SLEW"])
    actuator_mapper.reset()
    controller.reset_path_tracking()

    log_file = None
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log_path.open("w", encoding="utf-8")

    print(
        f"[Loop] dt={DT:.3f}s ({SIM_HZ} Hz physics, {STEPS_PER_CTRL} step(s)/tick), "
        f"max_steps={args.max_steps}, actuator_profile={ctrl_key}"
    )

    solve_times: list[float] = []
    trajectory: list[np.ndarray] = []
    try:
        for step in range(args.max_steps):
            state, vel_vec, _pos, sim_time_ms = poll_vehicle_state_4d(vehicle)
            trajectory.append(state.copy())

            t0 = time.perf_counter()
            u = controller.compute_action(
                jnp.asarray(state),
                rrt_path,
                wheel_base=WHEEL_BASE,
            )
            solve_times.append(time.perf_counter() - t0)

            throttle, brake, steering = actuator_mapper.map_control(
                u, state, vel_vec
            )

            vehicle.control(
                throttle=throttle,
                steering=steering,
                brake=brake,
                parkingbrake=0.0,
                gear=2,
            )
            bng.control.step(STEPS_PER_CTRL)

            if step % 10 == 0:
                pos_err = float(
                    np.hypot(state[0] - GOAL_POSITION[0], state[1] - GOAL_POSITION[1])
                )
                print(
                    f"  tick {step:4d} | pos=({state[0]:.1f}, {state[1]:.1f}) "
                    f"θ={state[2]:.2f} v={state[3]:.2f} | err={pos_err:.1f} m | "
                    f"thr={throttle:.2f} brk={brake:.2f} steer={steering:+.2f}",
                    flush=True,
                )

            if log_file is not None:
                pos_err = float(
                    np.hypot(state[0] - GOAL_POSITION[0], state[1] - GOAL_POSITION[1])
                )
                record = {
                    "tick": step,
                    "timestamp_ms": sim_time_ms,
                    "pos": [state[0], state[1], _pos[2]],
                    "velocity": list(vel_vec),
                    "state": state.tolist(),
                    "control_mpc": [float(u[0]), float(u[1])],
                    "throttle": throttle,
                    "brake": brake,
                    "steering": steering,
                    "goal_error_m": pos_err,
                    "solve_time_s": solve_times[-1],
                }
                log_file.write(json.dumps(record) + "\n")

            if goal_reached(state):
                pos_err = float(
                    np.hypot(state[0] - GOAL_POSITION[0], state[1] - GOAL_POSITION[1])
                )
                print(f"[Done] Goal reached at tick {step} (error={pos_err:.3f} m)")
                break
        else:
            state, _, _, _ = poll_vehicle_state_4d(vehicle)
            pos_err = float(
                np.hypot(state[0] - GOAL_POSITION[0], state[1] - GOAL_POSITION[1])
            )
            print(f"[Done] Max steps reached (goal error={pos_err:.2f} m)")
    finally:
        vehicle.control(throttle=0.0, steering=0.0, brake=1.0, parkingbrake=1.0, gear=2)
        if log_file is not None:
            log_file.close()
            print(f"[Log] Wrote telemetry to {args.log_path}")

    if args.profile and solve_times:
        arr = np.asarray(solve_times)
        print(
            f"[Profile] controller: mean {arr.mean():.3f}s/tick, "
            f"max {arr.max():.3f}s, total {arr.sum():.1f}s"
        )

    return np.asarray(trajectory) if trajectory else np.empty((0, 4)), solve_times


def main() -> None:
    args = parse_args()
    set_up_simple_logging()

    dynamics_name: DynamicsName = args.dynamics  # type: ignore[assignment]
    controller_key = normalize_controller_name(args.controller)
    use_sindy = dynamics_name == "sindy"
    dynamics = get_dynamics(dynamics_name)
    if use_sindy:
        from sindympc.dynamics.sindy_model import (
            kinematic_steer_coefficient,
            theta_steering_gain,
        )

        steer_c = kinematic_steer_coefficient()
        gain = theta_steering_gain()
        if steer_c is not None and abs(steer_c) < 0.5:
            print(
                f"[SINDy] weak θ̇ on v·tan(δ)/L (learned {steer_c:+.4f}, nominal +1.0); "
                f"MPC rollout gain={gain:.2f} (set SINDYMPC_SINDY_THETA_GAIN=1 to disable)"
            )

    target_speed = resolve_target_speed(
        args, controller_key=controller_key, use_sindy=use_sindy,
    )
    args.target_speed_mps = target_speed
    if args.max_steps is None:
        args.max_steps = compute_max_steps(_MAX_ROUTE_M, target_speed, DT)

    print(f"[Plan] RRT start={RRT_START}, goal={RRT_GOAL}, obstacles={len(OBSTACLE_LIST)}")
    rrt_path = run_rrt_planner()
    if rrt_path is None:
        print("RRT failed to find a path. Exiting.")
        sys.exit(1)

    print(f"[Plan] RRT path: {len(rrt_path)} waypoints")
    rrt_path = densify_polyline(
        smooth_path_moving_average(
            chaikin_smooth(
                smooth_path_moving_average(rrt_path, passes=PATH_SMOOTH_PASSES),
                iterations=PATH_CHAIKIN_ITERS,
            ),
            passes=1,
        ),
        spacing_m=PATH_SPACING_M,
    )
    print(f"[Plan] Reference path: {len(rrt_path)} waypoints (~{PATH_SPACING_M} m spacing)")

    controller = build_controller(
        controller_key,
        dynamics,
        use_sindy=use_sindy,
        target_speed=target_speed,
    )
    ctrl_label = "SQP-MPC" if controller_key == "sqp" else "MPPI"
    print(
        f"[Config] controller={ctrl_label}, dynamics={dynamics_name}, "
        f"target_speed={target_speed:.2f} m/s, max_steps={args.max_steps}, "
        f"actuator_profile={args.actuator_profile}"
    )

    spawn_rot = resolve_spawn_rotation(args, rrt_path)
    spawn_heading = path_start_heading(rrt_path)
    if args.spawn_heading is not None:
        spawn_heading = args.spawn_heading
    print(f"[Sim] Spawn heading={spawn_heading:.2f} rad ({np.degrees(spawn_heading):.0f}°)")

    print(f"[Init] Connecting to BeamNG at {args.host}:{args.port}")
    bng = BeamNGpy(args.host, args.port, home=args.bng_home, headless=args.headless)
    scenario, vehicle = build_beamng_scenario(bng, args, spawn_rot)

    executed_trajectory = np.empty((0, 4))
    solve_times: list[float] = []
    bng.open(launch=not args.no_launch)
    try:
        scenario.make(bng)
        bng.settings.set_deterministic(SIM_HZ)
        bng.scenario.load(scenario)
        bng.ui.hide_hud()
        bng.scenario.start()
        bng.control.pause()

        vehicle.queue_lua_command('controller.mainController.setGearboxMode("arcade")')

        print(f"[Sim] Warm-up: {args.warmup_steps} physics steps")
        vehicle.control(
            throttle=0.0, steering=0.0, brake=0.0, parkingbrake=0.0, gear=2,
        )
        for _ in range(args.warmup_steps):
            bng.control.step(1)
            vehicle.sensors.poll("state")

        spawn_state, _, spawn_pos, _ = poll_vehicle_state_4d(vehicle)
        spawn_err = float(
            np.hypot(spawn_state[0] - RRT_START[0], spawn_state[1] - RRT_START[1])
        )
        print(
            f"[Sim] Spawn XY=({spawn_state[0]:.1f}, {spawn_state[1]:.1f}, z={spawn_pos[2]:.1f}), "
            f"θ={spawn_state[2]:.2f} rad, offset from RRT start={spawn_err:.1f} m"
        )
        if spawn_err > 5.0:
            print(
                "[Warn] Vehicle spawn is far from RRT start — adjust --spawn-x/y/z or "
                "confirm --map matches beamng_defaults world frame."
            )

        executed_trajectory, solve_times = run_live_loop(
            bng, vehicle, controller, rrt_path, args,
        )
    finally:
        try:
            bng.ui.show_hud()
        except Exception:
            pass
        bng.close()

    summary = summarize_run(
        args.log_path,
        rrt_path,
        GOAL_POSITION,
        target_speed_mps=target_speed,
        goal_threshold_m=GOAL_THRESHOLD,
        goal_slow_radius_m=GOAL_SLOW_RADIUS,
        dt=DT,
        solve_times=solve_times if args.profile else None,
    )
    summary.update(
        {
            "controller": controller_key,
            "dynamics": dynamics_name,
        }
    )
    print(
        f"[Metrics] mean_v={summary['mean_speed_mps']:.2f} m/s, "
        f"final_err={summary['final_goal_error_m']:.2f} m, "
        f"mean_cte={summary['mean_cross_track_error_m']:.2f} m, "
        f"mean_thr={summary['mean_throttle']:.3f}, mean_a_mpc={summary['mean_mpc_accel']:.3f}"
    )

    if not args.no_plot:
        plot_path = args.plot_path or default_plot_path(args.controller, dynamics_name)
        plot_beamng_run(
            executed_trajectory=executed_trajectory,
            rrt_path=rrt_path,
            obstacle_list=OBSTACLE_LIST,
            start_xy=BEAMNG_START_XY,
            goal_xy=BEAMNG_GOAL_XY,
            controller=args.controller,
            dynamics=dynamics_name,
            save_path=plot_path,
            show=args.show_plot,
        )


if __name__ == "__main__":
    main()
