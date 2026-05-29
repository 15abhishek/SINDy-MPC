"""Compare kinematic bicycle vs pure SINDy on one-step and short rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sindympc.data.telemetry import load_aligned_telemetry
from sindympc.dynamics import get_dynamics
from sindympc.system_id.sindy_training import controls_for_library_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHEEL_BASE = 2.5


def one_step_rmse(
    plant,
    states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    n: int,
    *,
    sindy_plant: bool = False,
) -> dict[str, float]:
    errs_xy, errs_th, errs_v = [], [], []
    for k in range(n):
        x0 = jnp.asarray(states[k])
        u = jnp.array([controls[k, 0], controls[k, 1]])
        kwargs = dict(clip_state=False) if sindy_plant else {}
        x1_pred = plant.rk4_step(x0, u, WHEEL_BASE, dt, **kwargs)
        x1_true = jnp.asarray(states[k + 1])
        e = x1_pred - x1_true
        errs_xy.append(float(e[0] ** 2 + e[1] ** 2))
        errs_th.append(float(e[2] ** 2))
        errs_v.append(float(e[3] ** 2))
    return {
        "rmse_xy": float(np.sqrt(np.mean(errs_xy))),
        "rmse_theta": float(np.sqrt(np.mean(errs_th))),
        "rmse_v": float(np.sqrt(np.mean(errs_v))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    traj = PROJECT_ROOT / "data" / "trajectory.jsonl"
    ctrl = PROJECT_ROOT / "data" / "controls.jsonl"
    states, controls_log, timestamps_ms, dt = load_aligned_telemetry(traj, ctrl)
    controls = controls_for_library_mode(
        states, controls_log, timestamps_ms, "kinematic"
    )
    n = min(args.steps, len(states) - 1)

    bicycle = get_dynamics("bicycle")
    sindy = get_dynamics("sindy")

    u_mpc = np.column_stack([controls[:, 0], controls[:, 1]])

    print(f"[Compare] {n} one-step RK4 predictions, dt = {dt:.4f} s\n")
    b = one_step_rmse(bicycle, states, u_mpc, dt, n, sindy_plant=False)
    s = one_step_rmse(sindy, states, u_mpc, dt, n, sindy_plant=True)

    print(f"{'metric':<14} {'bicycle':>10} {'sindy':>10}")
    print("-" * 36)
    for key in ("rmse_xy", "rmse_theta", "rmse_v"):
        print(f"{key:<14} {b[key]:10.4f} {s[key]:10.4f}")


if __name__ == "__main__":
    main()
