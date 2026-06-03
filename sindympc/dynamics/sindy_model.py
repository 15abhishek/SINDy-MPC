"""Learned vehicle dynamics from SINDy coefficients (JAX, no PySINDy at runtime)."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np

from sindympc.config.beamng_defaults import (
    BEAMNG_CONTROL_HIGH,
    BEAMNG_CONTROL_LOW,
    BEAMNG_STATE_HIGH,
    BEAMNG_STATE_LOW,
)
from sindympc.system_id.sindy_features import (
    IDX_DELTA,
    IDX_THETA,
    IDX_U0,
    IDX_V,
    MAX_DELTA_FOR_TAN,
)

# Coefficients and feature metadata (see ``save_model_npz``).
_coefficients: Optional[jnp.ndarray] = None
_poly_powers: Optional[jnp.ndarray] = None
_n_poly_features: int = 0
_model_version: int = 1
_model_path: Optional[Path] = None
_poly_input_indices: Optional[jnp.ndarray] = None
_n_kinematic_features: int = 4
_n_residual_trig_features: int = 0
_wheel_base: float = 2.5
_uses_acceleration_control: bool = False
_theta_steering_gain: float = 1.0

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "sindy_model.npz"
)
LEGACY_PICKLE_PATH = DEFAULT_MODEL_PATH.with_suffix(".pkl")
MODEL_VERSION_BICYCLE = 2
MODEL_VERSION_BICYCLE_LOCAL_POLY = 3
MODEL_VERSION_KINEMATIC = 4  # legacy: throttle, v·tan(δ)
MODEL_VERSION_RESIDUAL_V2 = 5
MODEL_VERSION_KINEMATIC_A = 6  # v·tan(δ)/L and acceleration a (pure bicycle basis)


def save_model_npz(model: object, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    """Export fitted SINDy model for JAX MPC (polynomial + bicycle trig features)."""
    resolved = Path(path)
    fl = model.feature_library

    if hasattr(fl, "powers_"):
        # BicycleFeatureLibrary
        poly_powers = np.asarray(fl.powers_, dtype=np.int32)
        n_poly = int(poly_powers.shape[0])
        poly_input_indices = np.asarray(fl.poly_input_indices, dtype=np.int32)
        lib_mode = getattr(fl, "mode", "residual")
        n_kin = int(getattr(fl, "n_kinematic_features", 4))
        n_rt = int(getattr(fl, "n_residual_trig_features", 0))
        if lib_mode == "kinematic":
            version = (
                MODEL_VERSION_KINEMATIC_A
                if getattr(fl, "_use_acceleration", False)
                else MODEL_VERSION_KINEMATIC
            )
        elif n_rt > 0 or n_poly > 0:
            version = MODEL_VERSION_RESIDUAL_V2
        else:
            version = MODEL_VERSION_BICYCLE_LOCAL_POLY
    elif hasattr(fl, "libraries"):
        # Legacy ConcatLibrary fallback — first child polynomial
        poly_powers = np.asarray(fl.libraries[0].powers_, dtype=np.int32)
        n_poly = int(poly_powers.shape[0])
        version = 1
        poly_input_indices = np.arange(poly_powers.shape[1], dtype=np.int32)
        n_kin = 0
        n_rt = 0
    else:
        raise TypeError("Unsupported feature library for NPZ export.")

    wb = float(getattr(fl, "wheel_base", 2.5))

    np.savez(
        resolved,
        coefficients=np.asarray(model.coefficients(), dtype=np.float64),
        poly_powers=poly_powers,
        n_poly_features=np.array([n_poly], dtype=np.int32),
        model_version=np.array([version], dtype=np.int32),
        poly_input_indices=poly_input_indices,
        n_kinematic_features=np.array([n_kin], dtype=np.int32),
        n_residual_trig_features=np.array([n_rt], dtype=np.int32),
        wheel_base=np.array([wb], dtype=np.float64),
    )
    return resolved


def _load_npz(path: Path) -> None:
    global _coefficients, _poly_powers, _n_poly_features, _model_version, _model_path
    global _poly_input_indices, _n_kinematic_features, _n_residual_trig_features
    global _wheel_base, _uses_acceleration_control
    archive = np.load(path)
    _coefficients = jnp.asarray(archive["coefficients"])
    _model_version = int(np.asarray(archive.get("model_version", [1])).reshape(-1)[0])
    _poly_input_indices = (
        jnp.asarray(archive["poly_input_indices"])
        if "poly_input_indices" in archive
        else None
    )
    _n_kinematic_features = int(
        np.asarray(archive.get("n_kinematic_features", [4])).reshape(-1)[0]
    )
    _n_residual_trig_features = int(
        np.asarray(archive.get("n_residual_trig_features", [0])).reshape(-1)[0]
    )
    _wheel_base = float(np.asarray(archive.get("wheel_base", [2.5])).reshape(-1)[0])
    _uses_acceleration_control = _model_version == MODEL_VERSION_KINEMATIC_A
    if _model_version < MODEL_VERSION_RESIDUAL_V2:
        _n_residual_trig_features = 6 if _model_version >= MODEL_VERSION_BICYCLE else 0
        _n_kinematic_features = 0 if _model_version < MODEL_VERSION_RESIDUAL_V2 else 4

    if "poly_powers" in archive:
        _poly_powers = jnp.asarray(archive["poly_powers"])
        _n_poly_features = int(np.asarray(archive["n_poly_features"]).reshape(-1)[0])
    else:
        # Oldest format: ``powers`` only (pure cubic polynomial library)
        _poly_powers = jnp.asarray(archive["powers"])
        _n_poly_features = int(_poly_powers.shape[0])
        _model_version = 1

    _model_path = path
    _configure_theta_steering_gain()


def _kinematic_steer_feature_index() -> int:
    """Column index of ``v·tan(δ)/L`` in the feature vector (kinematic-A layout)."""
    return _n_poly_features + 2


def _configure_theta_steering_gain() -> None:
    """Optional scale on the learned ``v·tan(δ)/L`` term in θ̇ (MPC rollouts only)."""
    global _theta_steering_gain
    env_gain = os.environ.get("SINDYMPC_SINDY_THETA_GAIN")
    if env_gain is not None:
        _theta_steering_gain = float(np.clip(float(env_gain), -20.0, 20.0))
        return
    _theta_steering_gain = 1.0
    if os.environ.get("SINDYMPC_SINDY_AUTO_THETA_GAIN", "1").strip() in ("0", "false", "no"):
        return
    if _model_version != MODEL_VERSION_KINEMATIC_A or _coefficients is None:
        return
    idx = _kinematic_steer_feature_index()
    if idx >= _coefficients.shape[1]:
        return
    coeff = float(_coefficients[IDX_THETA, idx])
    if 0.05 < abs(coeff) < 0.95:
        # Effective coeff becomes gain * coeff; choose gain so gain * coeff ≈ +1 (nominal).
        _theta_steering_gain = float(np.clip(1.0 / coeff, -15.0, 15.0))


def theta_steering_gain() -> float:
    """Multiplier applied to the ``v·tan(δ)/L`` contribution in θ̇."""
    return _theta_steering_gain


def kinematic_steer_coefficient() -> Optional[float]:
    """Learned coefficient on ``v·tan(δ)/L`` in the θ̇ row (None if not applicable)."""
    if _coefficients is None or _model_version != MODEL_VERSION_KINEMATIC_A:
        return None
    idx = _kinematic_steer_feature_index()
    if idx >= _coefficients.shape[1]:
        return None
    return float(_coefficients[IDX_THETA, idx])


def load_model(path: Path | str | None = None) -> Path:
    """Load SINDy NPZ artifacts (re-export legacy ``.pkl`` if needed)."""
    if path is None:
        resolved = DEFAULT_MODEL_PATH
        if not resolved.is_file() and LEGACY_PICKLE_PATH.is_file():
            with LEGACY_PICKLE_PATH.open("rb") as f:
                save_model_npz(pickle.load(f), resolved)
    else:
        resolved = Path(path)

    if resolved.suffix == ".pkl":
        npz_path = resolved.with_suffix(".npz")
        if not npz_path.is_file():
            with resolved.open("rb") as f:
                save_model_npz(pickle.load(f), npz_path)
        resolved = npz_path

    if not resolved.is_file():
        legacy = (
            resolved.with_suffix(".pkl")
            if resolved.suffix == ".npz"
            else LEGACY_PICKLE_PATH
        )
        if legacy.is_file():
            with legacy.open("rb") as f:
                save_model_npz(pickle.load(f), resolved)
        else:
            raise FileNotFoundError(
                f"SINDy model not found: {resolved}\n"
                "Train with: python scripts/02_train_sindy.py"
            )

    _load_npz(resolved)
    return resolved


def model_path() -> Optional[Path]:
    return _model_path


def _require_artifacts() -> tuple[jnp.ndarray, jnp.ndarray, int, int]:
    if _coefficients is None or _poly_powers is None:
        raise RuntimeError(
            "SINDy model not loaded. Call load_model() or get_dynamics('sindy') first."
        )
    return _coefficients, _poly_powers, _n_poly_features, _model_version


_STATE_LOW = jnp.asarray(BEAMNG_STATE_LOW)
_STATE_HIGH = jnp.asarray(BEAMNG_STATE_HIGH)
_CONTROL_LOW = jnp.asarray(BEAMNG_CONTROL_LOW)
_CONTROL_HIGH = jnp.asarray(BEAMNG_CONTROL_HIGH)


def _map_mpc_control_to_training(control: jnp.ndarray) -> jnp.ndarray:
    """Map MPC ``[a, delta]`` to the control coordinates used at train time."""
    delta = jnp.clip(control[1], _CONTROL_LOW[1], _CONTROL_HIGH[1])
    if _uses_acceleration_control:
        a = jnp.clip(control[0], -1.0, 1.0)
        return jnp.array([a, delta])
    throttle = 0.5 * (control[0] + 1.0)
    throttle = jnp.clip(throttle, _CONTROL_LOW[0], _CONTROL_HIGH[0])
    return jnp.array([throttle, delta])


def _clip_state(state: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(state, _STATE_LOW, _STATE_HIGH)


def _preprocess_inputs(
    state: jnp.ndarray,
    control: jnp.ndarray,
    *,
    clip_state: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert MPC controls to training controls and (optionally) clip state.

    - In MPC/simulation we keep clipping ON for safety/stability.
    - In open-loop validation we may disable state clipping to avoid
      evaluating a clipped model against unclipped logs.
    """
    if clip_state:
        state = _clip_state(state)
    control = _map_mpc_control_to_training(control)
    control = jnp.clip(control, _CONTROL_LOW, _CONTROL_HIGH)
    return state, control


def _polynomial_features(xu: jnp.ndarray, powers: jnp.ndarray) -> jnp.ndarray:
    return jnp.prod(xu**powers, axis=1)


def _kinematic_bicycle_features(xu: jnp.ndarray, wheel_base: float) -> jnp.ndarray:
    """v cos θ, v sin θ, (v tan δ)/L, a — pure kinematic basis (model v6)."""
    theta = xu[IDX_THETA]
    v = xu[IDX_V]
    delta = xu[IDX_DELTA]
    u0 = xu[IDX_U0]
    tan_delta = jnp.tan(jnp.clip(delta, -MAX_DELTA_FOR_TAN, MAX_DELTA_FOR_TAN))
    inv_l = 1.0 / jnp.maximum(wheel_base, 1e-6)
    return jnp.array(
        [
            v * jnp.cos(theta),
            v * jnp.sin(theta),
            v * tan_delta * inv_l,
            u0,
        ]
    )


def _kinematic_legacy_features(xu: jnp.ndarray) -> jnp.ndarray:
    """Legacy v4: v cos θ, v sin θ, v tan δ, throttle."""
    theta = xu[IDX_THETA]
    v = xu[IDX_V]
    delta = xu[IDX_DELTA]
    throttle = xu[IDX_U0]
    tan_delta = jnp.tan(jnp.clip(delta, -MAX_DELTA_FOR_TAN, MAX_DELTA_FOR_TAN))
    return jnp.array(
        [
            v * jnp.cos(theta),
            v * jnp.sin(theta),
            v * tan_delta,
            throttle,
        ]
    )


def _legacy_bicycle_trig_features(xu: jnp.ndarray) -> jnp.ndarray:
    """Legacy v1–v3 library: sin θ, cos θ, tan δ, v sin θ, v cos θ, v tan δ."""
    theta = xu[IDX_THETA]
    v = xu[IDX_V]
    delta = xu[IDX_DELTA]
    tan_delta = jnp.tan(jnp.clip(delta, -MAX_DELTA_FOR_TAN, MAX_DELTA_FOR_TAN))
    sin_t = jnp.sin(theta)
    cos_t = jnp.cos(theta)
    return jnp.array(
        [
            sin_t,
            cos_t,
            tan_delta,
            v * sin_t,
            v * cos_t,
            v * tan_delta,
        ]
    )


def _residual_extra_trig_features(xu: jnp.ndarray) -> jnp.ndarray:
    """sin θ, cos θ, tan δ — residual library only (v2 layout)."""
    theta = xu[IDX_THETA]
    delta = xu[IDX_DELTA]
    tan_delta = jnp.tan(jnp.clip(delta, -MAX_DELTA_FOR_TAN, MAX_DELTA_FOR_TAN))
    return jnp.array([jnp.sin(theta), jnp.cos(theta), tan_delta])


def _feature_vector(
    state: jnp.ndarray,
    control: jnp.ndarray,
    *,
    clip_state: bool,
) -> jnp.ndarray:
    coefficients, powers, n_poly, version = _require_artifacts()
    state, control = _preprocess_inputs(state, control, clip_state=clip_state)
    xu = jnp.concatenate([state, control])

    if version == MODEL_VERSION_KINEMATIC_A:
        return _kinematic_bicycle_features(xu, _wheel_base)

    if version == MODEL_VERSION_KINEMATIC:
        return _kinematic_legacy_features(xu)

    if version >= MODEL_VERSION_RESIDUAL_V2:
        parts = []
        if n_poly > 0 and _poly_input_indices is not None:
            parts.append(_polynomial_features(xu[_poly_input_indices], powers))
        parts.append(_kinematic_legacy_features(xu))
        if _n_residual_trig_features > 0:
            parts.append(_residual_extra_trig_features(xu))
        return jnp.concatenate(parts)

    if version >= MODEL_VERSION_BICYCLE:
        if _poly_input_indices is not None and version >= MODEL_VERSION_BICYCLE_LOCAL_POLY:
            poly_in = xu[_poly_input_indices]
        else:
            poly_in = xu
        poly = _polynomial_features(poly_in, powers) if n_poly > 0 else jnp.array([])
        trig = _legacy_bicycle_trig_features(xu)
        return jnp.concatenate([poly, trig]) if n_poly > 0 else trig

    return _polynomial_features(xu, powers)


def continuous_dynamics(
    state: jnp.ndarray,
    control: jnp.ndarray,
    wheel_base: float,
    dt: float,
    *,
    clip_state: bool = True,
    control_frame: str = "mpc",
) -> jnp.ndarray:
    """Evaluate Ẋ = f(x, u) from exported SINDy coefficients (pure learned RHS)."""
    del wheel_base, dt, control_frame
    coefficients, _, _, version = _require_artifacts()
    features = _feature_vector(state, control, clip_state=clip_state)
    xdot = coefficients @ features
    gain = _theta_steering_gain
    if gain != 1.0 and version == MODEL_VERSION_KINEMATIC_A:
        idx = _kinematic_steer_feature_index()
        if idx < features.shape[0]:
            extra = (gain - 1.0) * coefficients[IDX_THETA, idx] * features[idx]
            xdot = xdot.at[IDX_THETA].add(extra)
    return xdot


def rk4_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    wheel_base: float,
    dt: float,
    *,
    clip_state: bool = True,
    control_frame: str = "mpc",
) -> jnp.ndarray:
    """RK4 integrate SINDy dynamics (same signature as :mod:`bicycle_model`)."""
    half_dt = 0.5 * dt
    kwargs = dict(clip_state=clip_state, control_frame=control_frame)
    f1 = continuous_dynamics(state, control, wheel_base, dt, **kwargs)
    f2 = continuous_dynamics(
        state + half_dt * f1, control, wheel_base, half_dt, **kwargs
    )
    f3 = continuous_dynamics(
        state + half_dt * f2, control, wheel_base, half_dt, **kwargs
    )
    f4 = continuous_dynamics(
        state + dt * f3, control, wheel_base, half_dt, **kwargs
    )
    next_state = state + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
    next_state = jnp.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)
    return _clip_state(next_state) if clip_state else next_state
