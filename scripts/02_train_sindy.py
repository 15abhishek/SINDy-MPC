"""Train a SINDy model on BeamNG telemetry and save to disk."""

from __future__ import annotations

from pathlib import Path

from sindympc.data.telemetry import load_aligned_telemetry
from sindympc.dynamics.sindy_model import save_model_npz
from sindympc.system_id.sindy_training import train_sindy_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_DATA_PATH = PROJECT_ROOT / "data" / "trajectory.jsonl"
CONTROL_DATA_PATH = PROJECT_ROOT / "data" / "controls.jsonl"
MODEL_PATH = PROJECT_ROOT / "data" / "sindy_model.npz"


def main() -> None:
    states, controls, dt = load_aligned_telemetry(
        TRAJECTORY_DATA_PATH, CONTROL_DATA_PATH
    )
    print(
        f"[Load] {len(states)} aligned samples, dt = {dt:.6f} s"
    )

    print(
        "[Train] Fitting SINDy (Savitzky–Golay + bicycle feature library "
        "+ STLSQ)..."
    )
    model = train_sindy_model(states, controls, dt)

    print("\n[Equations] Discovered dynamics (Ẋ = f(x, u)):")
    model.print()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model_npz(model, MODEL_PATH)
    print(f"\n[Save] Coefficients written to {MODEL_PATH} (NPZ, JAX-ready)")


if __name__ == "__main__":
    main()
