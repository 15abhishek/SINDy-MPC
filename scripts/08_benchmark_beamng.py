"""BeamNG closed-loop benchmark sweeps and offline open-loop speed stratification.

Usage
-----
# Offline open-loop RMSE by speed bin (no BeamNG):
python scripts/08_benchmark_beamng.py --offline-only

# Live BeamNG matrix (requires running BeamNG):
python scripts/08_benchmark_beamng.py --live \\
    --target-speeds 2,3.5,5 \\
    --controllers sqp,mppi \\
    --dynamics sindy,bicycle
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sindympc.config.beamng_defaults import (  # noqa: E402
    BEAMNG_GOAL_XY,
    BEAMNG_GOAL_SLOW_RADIUS_M,
    BEAMNG_GOAL_THRESHOLD_M,
    BEAMNG_START_XY,
    BEAMNG_DT,
)
from sindympc.data.telemetry import load_aligned_telemetry  # noqa: E402
from sindympc.dynamics import get_dynamics  # noqa: E402
from sindympc.system_id.sindy_training import controls_for_library_mode  # noqa: E402
from sindympc.utils.benchmark_metrics import (  # noqa: E402
    format_summary_markdown,
    summarize_run,
    write_summary_csv,
)

WHEEL_BASE = 2.5
_ROUTE_M = float(
    np.hypot(BEAMNG_GOAL_XY[0] - BEAMNG_START_XY[0], BEAMNG_GOAL_XY[1] - BEAMNG_START_XY[1])
)


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def one_step_rmse(
    plant,
    states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    mask: np.ndarray,
    *,
    sindy_plant: bool = False,
) -> dict[str, float]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return {"rmse_xy": float("nan"), "rmse_theta": float("nan"), "rmse_v": float("nan"), "n": 0}
    errs_xy, errs_th, errs_v = [], [], []
    for k in idx:
        if k + 1 >= len(states):
            continue
        import jax.numpy as jnp

        x0 = jnp.asarray(states[k])
        u = jnp.array([controls[k, 0], controls[k, 1]])
        kwargs = dict(clip_state=False) if sindy_plant else {}
        x1_pred = plant.rk4_step(x0, u, WHEEL_BASE, dt, **kwargs)
        x1_true = jnp.asarray(states[k + 1])
        e = x1_pred - x1_true
        errs_xy.append(float(e[0] ** 2 + e[1] ** 2))
        errs_th.append(float(e[2] ** 2))
        errs_v.append(float(e[3] ** 2))
    if not errs_xy:
        return {"rmse_xy": float("nan"), "rmse_theta": float("nan"), "rmse_v": float("nan"), "n": 0}
    return {
        "rmse_xy": float(np.sqrt(np.mean(errs_xy))),
        "rmse_theta": float(np.sqrt(np.mean(errs_th))),
        "rmse_v": float(np.sqrt(np.mean(errs_v))),
        "n": len(errs_xy),
    }


def run_offline_speed_bins(
    trajectory: Path,
    controls_path: Path,
    *,
    output_dir: Path,
) -> list[dict]:
    """Open-loop one-step RMSE stratified by |v| bins."""
    states, controls_log, timestamps_ms, dt = load_aligned_telemetry(
        trajectory, controls_path,
    )
    controls = controls_for_library_mode(
        states, controls_log, timestamps_ms, "kinematic",
    )
    u_mpc = np.column_stack([controls[:, 0], controls[:, 1]])
    bicycle = get_dynamics("bicycle")
    sindy = get_dynamics("sindy")

    bins = [
        ("low_v_lt2", states[:, 3] < 2.0),
        ("med_v_2_4", (states[:, 3] >= 2.0) & (states[:, 3] < 4.0)),
        ("high_v_ge4", states[:, 3] >= 4.0),
    ]
    rows: list[dict] = []
    for label, mask in bins:
        b = one_step_rmse(bicycle, states, u_mpc, dt, mask, sindy_plant=False)
        s = one_step_rmse(sindy, states, u_mpc, dt, mask, sindy_plant=True)
        row = {
            "mode": "offline_openloop",
            "speed_bin": label,
            "n_samples": b["n"],
            "bicycle_rmse_xy": b["rmse_xy"],
            "bicycle_rmse_theta": b["rmse_theta"],
            "bicycle_rmse_v": b["rmse_v"],
            "sindy_rmse_xy": s["rmse_xy"],
            "sindy_rmse_theta": s["rmse_theta"],
            "sindy_rmse_v": s["rmse_v"],
        }
        rows.append(row)

    out = output_dir / "offline_speed_bins.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[Offline] Wrote {out}")
    return rows


def run_live_cell(
    *,
    controller: str,
    dynamics: str,
    target_speed: float,
    run_dir: Path,
    no_launch: bool,
    profile: bool,
) -> dict:
    log_path = run_dir / "beamng_live.jsonl"
    plot_path = run_dir / "trajectory.png"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "07_run_beamng_sim.py"),
        "--controller",
        controller,
        "--dynamics",
        dynamics,
        "--target-speed",
        str(target_speed),
        "--log-path",
        str(log_path),
        "--plot-path",
        str(plot_path),
    ]
    if no_launch:
        cmd.append("--no-launch")
    if profile:
        cmd.append("--profile")

    print(f"[Live] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    summary_path = run_dir / "summary.json"

    # Reference path length unknown without re-planning; use straight chord for CTE baseline
    ref = [(BEAMNG_START_XY[0], BEAMNG_START_XY[1]), (BEAMNG_GOAL_XY[0], BEAMNG_GOAL_XY[1])]
    metrics = summarize_run(
        log_path if log_path.is_file() else None,
        ref,
        BEAMNG_GOAL_XY,
        target_speed_mps=target_speed,
        goal_threshold_m=BEAMNG_GOAL_THRESHOLD_M,
        goal_slow_radius_m=BEAMNG_GOAL_SLOW_RADIUS_M,
        dt=BEAMNG_DT,
    )
    row = {
        "controller": controller,
        "dynamics": dynamics,
        "target_speed_mps": target_speed,
        "exit_code": proc.returncode,
        "route_length_m": _ROUTE_M,
        **metrics,
    }
    summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def plot_speed_vs_accuracy(rows: list[dict], out_path: Path) -> None:
    """Poster plot: target speed vs error and achieved speed."""
    live = [r for r in rows if "controller" in r and r.get("mode") != "offline_openloop"]
    if not live:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ctrl in sorted({r["controller"] for r in live}):
        for dyn in sorted({r["dynamics"] for r in live}):
            subset = [
                r for r in live if r["controller"] == ctrl and r["dynamics"] == dyn
            ]
            if not subset:
                continue
            subset = sorted(subset, key=lambda r: r["target_speed_mps"])
            xs = [r["target_speed_mps"] for r in subset]
            err = [r["final_goal_error_m"] for r in subset]
            mv = [r["mean_speed_mps"] for r in subset]
            label = f"{ctrl}+{dyn}"
            axes[0].plot(xs, err, "o-", label=label)
            axes[1].plot(xs, mv, "o-", label=label)
            axes[1].plot(xs, xs, "k--", alpha=0.3)

    axes[0].set_xlabel("Target speed [m/s]")
    axes[0].set_ylabel("Final goal error [m]")
    axes[0].set_title("Speed vs accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Target speed [m/s]")
    axes[1].set_ylabel("Mean speed [m/s]")
    axes[1].set_title("Speed tracking")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-speeds",
        default="2,3.5,5",
        help="Comma-separated cruise speeds [m/s] for live sweep",
    )
    parser.add_argument(
        "--controllers",
        default="sqp,mppi",
        help="Comma-separated: sqp, mppi",
    )
    parser.add_argument(
        "--dynamics",
        dest="dynamics_list",
        default="sindy,bicycle",
        help="Comma-separated: sindy, bicycle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "benchmark",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=PROJECT_ROOT / "data" / "trajectory.jsonl",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=PROJECT_ROOT / "data" / "controls.jsonl",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live BeamNG closed-loop matrix via scripts/07",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only run open-loop speed-bin analysis (default if --live not set)",
    )
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = args.output_dir / run_id
    bench_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    if args.offline_only or not args.live:
        offline_rows = run_offline_speed_bins(
            args.trajectory, args.controls, output_dir=bench_dir,
        )
        all_rows.extend(offline_rows)

    if args.live:
        speeds = parse_float_list(args.target_speeds)
        controllers = parse_list(args.controllers)
        dynamics = parse_list(args.dynamics_list)
        for ts in speeds:
            for ctrl in controllers:
                for dyn in dynamics:
                    cell_dir = bench_dir / f"{ctrl}_{dyn}_{ts:.1f}mps"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    row = run_live_cell(
                        controller=ctrl,
                        dynamics=dyn,
                        target_speed=ts,
                        run_dir=cell_dir,
                        no_launch=args.no_launch,
                        profile=args.profile,
                    )
                    all_rows.append(row)

    csv_path = args.output_dir / "summary.csv"
    write_summary_csv(all_rows, csv_path)
    print(f"[CSV] Wrote {csv_path}")

    live_rows = [r for r in all_rows if r.get("controller")]
    offline_rows = [r for r in all_rows if r.get("mode") == "offline_openloop"]
    plot_speed_vs_accuracy(live_rows, args.output_dir / "speed_vs_accuracy.png")

    print("\n## Benchmark summary\n")
    if offline_rows:
        print("### Open-loop speed bins\n")
        print(format_summary_markdown(offline_rows))
    if live_rows:
        if offline_rows:
            print("\n### Closed-loop live runs\n")
        print(format_summary_markdown(live_rows))
    if not offline_rows and not live_rows:
        print("_No benchmark runs._")


if __name__ == "__main__":
    main()
