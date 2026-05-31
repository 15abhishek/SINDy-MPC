"""Open-loop SINDy rollout vs logged BeamNG states (model check before MPC)."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sindympc.data.telemetry import load_aligned_telemetry
from sindympc.dynamics import get_dynamics
from sindympc.system_id.sindy_training import controls_for_library_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ = PROJECT_ROOT / "data" / "trajectory.jsonl"
DEFAULT_CTRL = PROJECT_ROOT / "data" / "controls.jsonl"
WHEEL_BASE = 2.5


def training_controls_to_mpc(controls: np.ndarray) -> np.ndarray:
    """Kinematic models use ``[a, delta]`` (same as MPC). Legacy throttle maps to ``a``."""
    from sindympc.dynamics.sindy_model import _uses_acceleration_control

    if _uses_acceleration_control:
        return np.asarray(controls, dtype=np.float64)
    mpc = np.empty_like(controls)
    mpc[:, 0] = 2.0 * controls[:, 0] - 1.0
    mpc[:, 1] = controls[:, 1]
    return mpc


def rollout_rmse(
    states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    *,
    n_steps: int | None = None,
    start: int = 0,
) -> tuple[float, np.ndarray]:
    """Free-rollout RMSE (predicted state is fed back in)."""
    dynamics = get_dynamics("sindy")
    u_mpc = training_controls_to_mpc(controls)
    end = len(states) - 1 if n_steps is None else min(start + n_steps, len(states) - 1)

    errs: list[float] = []
    x = jnp.asarray(states[start])
    for k in range(start, end):
        u = jnp.asarray(u_mpc[k])
        x_pred = dynamics.rk4_step(
            x, u, WHEEL_BASE, dt, clip_state=False, control_frame="training"
        )
        x_true = jnp.asarray(states[k + 1])
        errs.append(float(jnp.linalg.norm(x_pred - x_true)))
        x = x_pred

    err_arr = np.asarray(errs, dtype=np.float64)
    return float(np.sqrt(np.mean(err_arr**2))), err_arr


def teacher_forced_rmse(
    states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    *,
    n_steps: int | None = None,
    start: int = 0,
) -> dict[str, float]:
    """One-step RMSE when each prediction starts from the *true* state.

    This isolates model one-step accuracy from compounding rollout drift.
    """
    dynamics = get_dynamics("sindy")
    u_mpc = training_controls_to_mpc(controls)
    end = len(states) - 1 if n_steps is None else min(start + n_steps, len(states) - 1)

    exy2: list[float] = []
    etheta2: list[float] = []
    ev2: list[float] = []
    efull2: list[float] = []
    for k in range(start, end):
        x0 = jnp.asarray(states[k])
        u = jnp.asarray(u_mpc[k])
        x_pred = dynamics.rk4_step(
            x0, u, WHEEL_BASE, dt, clip_state=False, control_frame="training"
        )
        x_true = jnp.asarray(states[k + 1])
        e = x_pred - x_true

        exy2.append(float(e[0] ** 2 + e[1] ** 2))
        etheta2.append(float(e[2] ** 2))
        ev2.append(float(e[3] ** 2))
        efull2.append(float(jnp.dot(e, e)))

    def _rmse(v2: list[float]) -> float:
        return float(np.sqrt(np.mean(np.asarray(v2, dtype=np.float64))))

    return {
        "rmse_xy_m": _rmse(exy2),
        "rmse_theta_rad": _rmse(etheta2),
        "rmse_v_mps": _rmse(ev2),
        "rmse_full": _rmse(efull2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CTRL)
    parser.add_argument("--steps", type=int, default=500, help="Rollout length")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help=(
            "Start index in log. If 0, will auto-pick the first index whose (x,y) "
            "lies inside the BeamNG clamp corridor (matches MPC eval region)."
        ),
    )
    args = parser.parse_args()

    states, controls_log, timestamps_ms, dt = load_aligned_telemetry(
        args.trajectory, args.controls
    )
    controls = controls_for_library_mode(
        states, controls_log, timestamps_ms, "kinematic"
    )
    if args.start == 0:
        x, y = states[:, 0], states[:, 1]
        from sindympc.config.beamng_defaults import BEAMNG_STATE_HIGH, BEAMNG_STATE_LOW

        in_corridor = (
            (x >= BEAMNG_STATE_LOW[0])
            & (x <= BEAMNG_STATE_HIGH[0])
            & (y >= BEAMNG_STATE_LOW[1])
            & (y <= BEAMNG_STATE_HIGH[1])
        )
        # Find a start index whose entire validation window stays inside corridor,
        # otherwise the dynamics clamp will dominate the error metric.
        window = int(args.steps)
        start_idx = None
        for s in range(0, max(0, len(states) - window - 1)):
            if bool(np.all(in_corridor[s : s + window + 1])):
                start_idx = s
                break
        if start_idx is not None:
            args.start = int(start_idx)
            print(
                f"[Auto-start] Using start index {args.start} "
                f"(entire {args.steps}-step window inside clamp corridor)"
            )
        else:
            print(
                "[Auto-start] No full window stays inside clamp corridor; "
                "set --start manually (or widen clamps) to get meaningful metrics."
            )
    tf = teacher_forced_rmse(states, controls, dt, n_steps=args.steps, start=args.start)
    fr, _ = rollout_rmse(states, controls, dt, n_steps=args.steps, start=args.start)

    print(f"[One-step | teacher-forced] over {args.steps} steps")
    print(f"  RMSE_xy    : {tf['rmse_xy_m']:.4f} m")
    print(f"  RMSE_theta : {tf['rmse_theta_rad']:.4f} rad")
    print(f"  RMSE_v     : {tf['rmse_v_mps']:.4f} m/s")
    print(f"  RMSE_full  : {tf['rmse_full']:.4f}")
    print(f"[Free rollout] RMSE_full over {args.steps} steps: {fr:.4f}")


if __name__ == "__main__":
    main()
