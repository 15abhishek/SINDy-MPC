"""Train a SINDy model on BeamNG telemetry and save to disk."""

from __future__ import annotations

import argparse
from pathlib import Path

from sindympc.data.telemetry import HeadingConvention, load_aligned_telemetry
from sindympc.dynamics.sindy_model import save_model_npz
from sindympc.system_id.sindy_features import LibraryMode
from sindympc.system_id.sindy_training import (
    DEFAULT_WHEEL_BASE,
    audit_kinematic_sign_consistency,
    control_names_for_mode,
    controls_for_library_mode,
    subsample_training_data,
    train_sindy_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_DATA_PATH = PROJECT_ROOT / "data" / "trajectory.jsonl"
CONTROL_DATA_PATH = PROJECT_ROOT / "data" / "controls.jsonl"
MODEL_PATH = PROJECT_ROOT / "data" / "sindy_model.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SINDy on BeamNG logs. Use --library kinematic for a model "
            "comparable to the JAX bicycle plant; control dataset size with "
            "--max-samples / --fraction / --stride. Effective dt is recomputed "
            "after subsampling."
        )
    )
    parser.add_argument(
        "--library",
        choices=("kinematic", "residual"),
        default="kinematic",
        help="kinematic: bicycle ODE terms only; residual: + polynomials & extras",
    )
    parser.add_argument(
        "--heading",
        choices=("velocity", "quaternion"),
        default="velocity",
        help=(
            "velocity: θ = atan2(vy, vx) when moving (matches bicycle); "
            "quaternion: raw BeamNG yaw"
        ),
    )
    parser.add_argument("--trajectory", type=Path, default=TRAJECTORY_DATA_PATH)
    parser.add_argument("--controls", type=Path, default=CONTROL_DATA_PATH)
    parser.add_argument("-o", "--output", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap training rows (evenly spaced if downsampling)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        metavar="F",
        help="Use first F fraction of sliced data, e.g. 0.1 for 10%%",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every STRIDE-th sample after --start/--end",
    )
    parser.add_argument("--start", type=int, default=0, help="First row index")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last row index (exclusive)",
    )
    parser.add_argument(
        "--wheel-base",
        type=float,
        default=DEFAULT_WHEEL_BASE,
        help="L [m] for bicycle comparison printout (θ̇ coeff ≈ 1/L)",
    )
    parser.add_argument(
        "--stlsq-threshold",
        type=float,
        default=None,
        help="Override STLSQ sparsity threshold",
    )
    parser.add_argument(
        "--no-sign-audit",
        action="store_true",
        help="Skip pre-training kinematic sign consistency check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library_mode: LibraryMode = args.library
    heading: HeadingConvention = args.heading

    states, controls, timestamps_ms, dt_base = load_aligned_telemetry(
        args.trajectory, args.controls, heading=heading
    )
    print(
        f"[Load] {len(states)} aligned samples, base dt = {dt_base:.6f} s, "
        f"heading = {heading}"
    )

    states, controls_log, timestamps_ms, info = subsample_training_data(
        states,
        controls,
        timestamps_ms,
        max_samples=args.max_samples,
        stride=args.stride,
        start=args.start,
        end=args.end,
        fraction=args.fraction,
    )
    controls = controls_for_library_mode(
        states, controls_log, timestamps_ms, library_mode
    )
    dt = info["dt"]
    print(
        f"[Subset] using {info['n_used']} / {info['n_total']} samples "
        f"(stride={info['stride']}, fraction={info['fraction']}, "
        f"max_samples={info['max_samples']})"
    )
    print(
        f"[Timing] effective dt = {dt:.6f} s "
        f"(base {info['dt_base']:.6f} s), "
        f"SG window = {info['sg_window_length']} samples "
        f"(~{info['sg_window_length'] * dt:.2f} s)"
    )

    if not args.no_sign_audit:
        audit_kinematic_sign_consistency(
            states, timestamps_ms, wheel_base=args.wheel_base
        )

    u_names = control_names_for_mode(library_mode)
    print(
        f"[Train] library={library_mode}, controls=({u_names[0]}, {u_names[1]}), "
        f"Savitzky–Golay + STLSQ ..."
    )
    model = train_sindy_model(
        states,
        controls,
        dt,
        library_mode=library_mode,
        wheel_base=args.wheel_base,
        stlsq_threshold=args.stlsq_threshold,
    )

    print("\n[Equations] Discovered dynamics (Ẋ = f(x, u)):")
    model.print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model_npz(model, args.output)
    print(f"\n[Save] Coefficients written to {args.output} (NPZ, JAX-ready)")


if __name__ == "__main__":
    main()
