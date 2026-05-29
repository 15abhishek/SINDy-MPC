"""PySINDy training for vehicle dynamics (library module — no script entry point)."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pysindy as ps
from pysindy.differentiation import SINDyDerivative

from sindympc.system_id.sindy_features import (
    KINEMATIC_FEATURE_NAMES,
    BicycleFeatureLibrary,
    LibraryMode,
)
from sindympc.data.telemetry import compute_median_dt

# State / control column semantics for BeamNG telemetry
STATE_NAMES: Tuple[str, ...] = ("x", "y", "theta", "v")
N_STATES = len(STATE_NAMES)

# Residual library: throttle from logs. Kinematic library: acceleration (MPC-compatible).
CONTROL_NAMES_THROTTLE: Tuple[str, ...] = ("throttle", "delta")
CONTROL_NAMES_ACCEL: Tuple[str, ...] = ("a", "delta")
CONTROL_NAMES: Tuple[str, ...] = CONTROL_NAMES_THROTTLE
N_CONTROLS = 2

# Savitzky–Golay: ~1 s physical window; sample count scales with ``dt``
SG_TARGET_WINDOW_S = 1.0
SG_MIN_WINDOW_LENGTH = 5
SG_POLYORDER = 3

# STLSQ sparsity
STLSQ_THRESHOLD_RESIDUAL = 0.02
STLSQ_THRESHOLD_KINEMATIC = 0.002 # was 0.002
STLSQ_ALPHA = 0.05

POLYNOMIAL_DEGREE_RESIDUAL = 2
DEFAULT_LIBRARY_MODE: LibraryMode = "kinematic"
DEFAULT_WHEEL_BASE = 2.5


def control_names_for_mode(mode: LibraryMode) -> Tuple[str, ...]:
    return CONTROL_NAMES_ACCEL if mode == "kinematic" else CONTROL_NAMES_THROTTLE


def controls_for_library_mode(
    states: np.ndarray,
    controls_log: np.ndarray,
    timestamps_ms: np.ndarray,
    mode: LibraryMode,
) -> np.ndarray:
    """Build SINDYc controls: kinematic uses estimated ``a``; residual keeps throttle."""
    controls_log = np.asarray(controls_log, dtype=np.float64)
    if mode == "kinematic":
        a = estimate_acceleration(states, timestamps_ms)
        return np.column_stack([a, controls_log[:, 1]])
    return controls_log


def build_feature_library(
    mode: LibraryMode = DEFAULT_LIBRARY_MODE,
    poly_degree: int = POLYNOMIAL_DEGREE_RESIDUAL,
    wheel_base: float = DEFAULT_WHEEL_BASE,
) -> BicycleFeatureLibrary:
    """Build Θ(x, u) for the chosen training mode."""
    return BicycleFeatureLibrary(
        mode=mode, poly_degree=poly_degree, wheel_base=wheel_base
    )


def estimate_acceleration(states: np.ndarray, timestamps_ms: np.ndarray) -> np.ndarray:
    from sindympc.data.telemetry import estimate_acceleration as _est

    return _est(states, timestamps_ms)


def savitzky_window_length(dt: float, target_window_s: float = SG_TARGET_WINDOW_S) -> int:
    """Odd Savitzky–Golay window length for uniform sample period ``dt`` [s]."""
    length = int(round(target_window_s / dt))
    if length % 2 == 0:
        length += 1
    return max(SG_MIN_WINDOW_LENGTH, length)


def subsample_training_data(
    states: np.ndarray,
    controls: np.ndarray,
    timestamps_ms: np.ndarray,
    *,
    max_samples: Optional[int] = None,
    stride: int = 1,
    start: int = 0,
    end: Optional[int] = None,
    fraction: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Select a subset of samples and compute the **effective** ``dt``.

    When ``stride > 1`` or ``max_samples`` downsamples rows, the median time
    between kept samples increases. The returned ``info['dt']`` reflects that
    so PySINDy differentiation uses the correct spacing.

    Returns:
        ``(states, controls, timestamps_ms, info)`` with ``info['dt']``.
    """
    states = np.asarray(states, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64).reshape(-1)
    n = len(states)
    if len(controls) != n or len(timestamps_ms) != n:
        raise ValueError("states, controls, and timestamps_ms must have equal length.")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if start < 0 or start >= n:
        raise ValueError(f"start={start} out of range for N={n}")
    stop = n if end is None else min(end, n)
    if stop <= start:
        raise ValueError(f"empty slice: start={start}, end={stop}")

    indices = np.arange(start, stop, stride, dtype=np.int64)

    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        keep = max(1, int(round(len(indices) * fraction)))
        indices = indices[:keep]

    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        if len(indices) > max_samples:
            pick = np.linspace(0, len(indices) - 1, max_samples, dtype=int)
            indices = indices[pick]

    ts_sub = timestamps_ms[indices]
    dt_eff = compute_median_dt(ts_sub)
    sg_len = savitzky_window_length(dt_eff)

    info = {
        "n_total": n,
        "n_used": len(indices),
        "start": start,
        "end": stop,
        "stride": stride,
        "fraction": fraction,
        "max_samples": max_samples,
        "indices": indices,
        "dt": dt_eff,
        "dt_base": compute_median_dt(timestamps_ms),
        "sg_window_length": sg_len,
    }
    return states[indices], controls[indices], ts_sub, info


def kinematic_bicycle_targets(wheel_base: float = DEFAULT_WHEEL_BASE) -> dict:
    """Nominal coefficients on the kinematic feature basis (matches ``bicycle_model``)."""
    return {
        "x": {"v cos(theta)": 1.0},
        "y": {"v sin(theta)": 1.0},
        "theta": {"v tan(delta) / L": 1.0},
        "v": {"a": 1.0},
    }


def audit_kinematic_sign_consistency(
    states: np.ndarray,
    timestamps_ms: np.ndarray,
    *,
    wheel_base: float = DEFAULT_WHEEL_BASE,
    min_speed: float = 0.5,
) -> dict:
    """Check whether finite-difference ẋ, ẏ align with v·cos θ and v·sin θ.

    Scans constant heading offsets to detect quaternion-vs-velocity frame mismatch
    (common cause of swapped sin/cos signs in SINDy vs bicycle).
    """
    states = np.asarray(states, dtype=np.float64)
    ts = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    dt_steps = np.diff(ts) / 1000.0
    xdot = np.diff(states[:, 0]) / dt_steps
    ydot = np.diff(states[:, 1]) / dt_steps
    theta = states[:-1, 2]
    v = states[:-1, 3]
    moving = v >= min_speed

    if not np.any(moving):
        return {"warning": "insufficient speed for sign audit"}

    offsets = np.linspace(-np.pi, np.pi, 25)
    best = {"offset_rad": 0.0, "score": -np.inf}
    for offset in offsets:
        th = theta + offset
        pred_x = v * np.cos(th)
        pred_y = v * np.sin(th)
        corr_x = float(np.corrcoef(xdot[moving], pred_x[moving])[0, 1])
        corr_y = float(np.corrcoef(ydot[moving], pred_y[moving])[0, 1])
        score = corr_x + corr_y
        if score > best["score"]:
            best = {
                "offset_rad": float(offset),
                "offset_deg": float(np.degrees(offset)),
                "corr_xdot_vcos": corr_x,
                "corr_ydot_vsin": corr_y,
                "score": score,
            }

    print("\n[Sign audit] Kinematic bicycle vs logged motion (moving samples only)")
    print(
        f"  best heading offset: {best['offset_deg']:+.1f}°  "
        f"(corr ẋ↔v cos θ = {best['corr_xdot_vcos']:.3f}, "
        f"corr ẏ↔v sin θ = {best['corr_ydot_vsin']:.3f})"
    )
    if abs(best["offset_deg"]) > 15.0:
        print(
            "  → large offset: use heading=velocity when loading telemetry "
            "(default) or apply a constant θ shift before training."
        )
    elif best["corr_xdot_vcos"] > 0.95 and best["corr_ydot_vsin"] > 0.95:
        print("  → heading convention matches kinematic bicycle (good).")
    else:
        print("  → moderate mismatch; check steering sign or low-speed samples.")

    return best


def print_bicycle_comparison(
    model: ps.SINDy,
    wheel_base: float = DEFAULT_WHEEL_BASE,
) -> None:
    """Print learned vs nominal kinematic bicycle coefficients side by side."""
    names = model.get_feature_names()
    coef = model.coefficients()
    targets = kinematic_bicycle_targets(wheel_base)
    state_labels = list(STATE_NAMES)

    print("\n[Compare] Learned SINDy vs nominal kinematic bicycle")
    print(f"  wheel_base L = {wheel_base:.3f} m  (θ̇ feature is v·tan(δ)/L, target coeff = 1)")
    for i, state in enumerate(state_labels):
        print(f"\n  ({state})':")
        row = coef[i]
        active = [(names[j], row[j]) for j in range(len(names)) if abs(row[j]) > 1e-6]
        if active:
            for feat, val in sorted(active, key=lambda t: -abs(t[1])):
                print(f"    learned: {val:+.4f} * {feat}")
        else:
            print("    learned: (no active terms)")

        nominal = targets.get(state, {})
        for feat, val in nominal.items():
            if feat in names:
                j = names.index(feat)
                learned = row[j]
                print(f"    nominal: {val:+.4f} * {feat}  (learned {learned:+.4f})")


def train_sindy_model(
    states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    *,
    library_mode: LibraryMode = DEFAULT_LIBRARY_MODE,
    wheel_base: float = DEFAULT_WHEEL_BASE,
    stlsq_threshold: Optional[float] = None,
) -> ps.SINDy:
    """Fit sparse SINDy (SINDYc) on BeamNG ``[x, y, θ, v]`` with ``[throttle, δ]``.

    Args:
        states: Shape ``(n_samples, 4)``.
        controls: Shape ``(n_samples, 2)`` — throttle, steering ``delta``.
        dt: Sample period [s].
        library_mode: ``"kinematic"`` (bicycle-comparable) or ``"residual"``.
        wheel_base: Used only for post-train comparison printout.
        stlsq_threshold: Override STLSQ threshold; default depends on mode.

    Returns:
        Fitted :class:`pysindy.SINDy` model.
    """
    states = np.asarray(states, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != N_STATES:
        raise ValueError(f"Expected states shape (n, {N_STATES}), got {states.shape}")
    if controls.ndim != 2 or controls.shape[1] != N_CONTROLS:
        raise ValueError(f"Expected controls shape (n, 2), got {controls.shape}")
    if len(states) != len(controls):
        raise ValueError("states and controls must have the same number of rows.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    sg_window = savitzky_window_length(dt)
    if len(states) < sg_window:
        raise ValueError(
            f"Need at least {sg_window} samples for Savitzky–Golay at dt={dt:.4f}s; "
            f"got {len(states)}"
        )

    sg_half_width = (sg_window - 1) // 2
    differentiation_method = SINDyDerivative(
        kind="savitzky_golay",
        order=SG_POLYORDER,
        left=sg_half_width,
        right=sg_half_width,
        iwindow=True,
    )

    if stlsq_threshold is None:
        stlsq_threshold = (
            STLSQ_THRESHOLD_KINEMATIC
            if library_mode == "kinematic"
            else STLSQ_THRESHOLD_RESIDUAL
        )

    feature_library = build_feature_library(
        mode=library_mode, wheel_base=wheel_base
    )
    optimizer = ps.STLSQ(threshold=stlsq_threshold, alpha=STLSQ_ALPHA)
    u_names = list(STATE_NAMES) + list(control_names_for_mode(library_mode))

    model = ps.SINDy(
        differentiation_method=differentiation_method,
        feature_library=feature_library,
        optimizer=optimizer,
    )

    model.fit(x=states, u=controls, t=dt, feature_names=u_names)

    if library_mode == "kinematic":
        print_bicycle_comparison(model, wheel_base=wheel_base)

    return model
