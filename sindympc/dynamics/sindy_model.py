"""Learned vehicle dynamics from SINDy coefficients (JAX, no PySINDy at runtime)."""

from __future__ import annotations

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

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "sindy_model.npz"
)
LEGACY_PICKLE_PATH = DEFAULT_MODEL_PATH.with_suffix(".pkl")
MODEL_VERSION_BICYCLE = 2
MODEL_VERSION_BICYCLE_LOCAL_POLY = 3


def save_model_npz(model: object, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    """Export fitted SINDy model for JAX MPC (polynomial + bicycle trig features)."""
    resolved = Path(path)
    fl = model.feature_library

    if hasattr(fl, "powers_"):
        # BicycleFeatureLibrary
        poly_powers = np.asarray(fl.powers_, dtype=np.int32)
        n_poly = int(poly_powers.shape[0])
        poly_input_indices = np.asarray(
            getattr(fl, "poly_input_indices", np.arange(poly_powers.shape[1])),
            dtype=np.int32,
        )
        version = (
            MODEL_VERSION_BICYCLE_LOCAL_POLY
            if poly_input_indices.shape[0] != poly_powers.shape[1]
            or not np.array_equal(poly_input_indices, np.arange(poly_powers.shape[1]))
            else MODEL_VERSION_BICYCLE
        )
    elif hasattr(fl, "libraries"):
        # Legacy ConcatLibrary fallback — first child polynomial
        poly_powers = np.asarray(fl.libraries[0].powers_, dtype=np.int32)
        n_poly = int(poly_powers.shape[0])
        version = 1
        poly_input_indices = np.arange(poly_powers.shape[1], dtype=np.int32)
    else:
        raise TypeError("Unsupported feature library for NPZ export.")

    np.savez(
        resolved,
        coefficients=np.asarray(model.coefficients(), dtype=np.float64),
        poly_powers=poly_powers,
        n_poly_features=np.array([n_poly], dtype=np.int32),
        model_version=np.array([version], dtype=np.int32),
        poly_input_indices=poly_input_indices,
    )
    return resolved


def _load_npz(path: Path) -> None:
    global _coefficients, _poly_powers, _n_poly_features, _model_version, _model_path, _poly_input_indices
    archive = np.load(path)
    _coefficients = jnp.asarray(archive["coefficients"])
    _model_version = int(np.asarray(archive.get("model_version", [1])).reshape(-1)[0])
    _poly_input_indices = (
        jnp.asarray(archive["poly_input_indices"])
        if "poly_input_indices" in archive
        else None
    )

    if "poly_powers" in archive:
        _poly_powers = jnp.asarray(archive["poly_powers"])
        _n_poly_features = int(np.asarray(archive["n_poly_features"]).reshape(-1)[0])
    else:
        # Oldest format: ``powers`` only (pure cubic polynomial library)
        _poly_powers = jnp.asarray(archive["powers"])
        _n_poly_features = int(_poly_powers.shape[0])
        _model_version = 1

    _model_path = path


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
    """Map MPC ``[a, delta]`` to SINDy training inputs ``[throttle, delta]``.

    Training data uses BeamNG **throttle** in ``[0, 1]``. MPC uses signed
    **acceleration** in ``[-1, 1]``. Without this map, negative ``a`` is clipped
    to zero throttle and positive ``a`` is misinterpreted.
    """
    throttle = 0.5 * (control[0] + 1.0)
    delta = jnp.clip(control[1], _CONTROL_LOW[1], _CONTROL_HIGH[1])
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


def _bicycle_trig_features(xu: jnp.ndarray) -> jnp.ndarray:
    """sin θ, cos θ, tan δ, v sin θ, v cos θ, v tan δ — matches training library."""
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


def _feature_vector(
    state: jnp.ndarray,
    control: jnp.ndarray,
    *,
    clip_state: bool,
) -> jnp.ndarray:
    coefficients, powers, n_poly, version = _require_artifacts()
    state, control = _preprocess_inputs(state, control, clip_state=clip_state)
    xu = jnp.concatenate([state, control])

    if version >= MODEL_VERSION_BICYCLE:
        if _poly_input_indices is not None and version >= MODEL_VERSION_BICYCLE_LOCAL_POLY:
            poly_in = xu[_poly_input_indices]
        else:
            poly_in = xu
        poly = _polynomial_features(poly_in, powers)
        trig = _bicycle_trig_features(xu)
        return jnp.concatenate([poly, trig])

    return _polynomial_features(xu, powers)


def continuous_dynamics(
    state: jnp.ndarray,
    control: jnp.ndarray,
    wheel_base: float,
    dt: float,
    *,
    clip_state: bool = True,
) -> jnp.ndarray:
    """Evaluate Ẋ = f(x, u) from exported SINDy coefficients."""
    del wheel_base, dt
    coefficients, _, _, _ = _require_artifacts()
    features = _feature_vector(state, control, clip_state=clip_state)
    return coefficients @ features


def rk4_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    wheel_base: float,
    dt: float,
    *,
    clip_state: bool = True,
) -> jnp.ndarray:
    """RK4 integrate SINDy dynamics (same signature as :mod:`bicycle_model`)."""
    half_dt = 0.5 * dt
    f1 = continuous_dynamics(state, control, wheel_base, dt, clip_state=clip_state)
    f2 = continuous_dynamics(
        state + half_dt * f1, control, wheel_base, half_dt, clip_state=clip_state
    )
    f3 = continuous_dynamics(
        state + half_dt * f2, control, wheel_base, half_dt, clip_state=clip_state
    )
    f4 = continuous_dynamics(
        state + dt * f3, control, wheel_base, half_dt, clip_state=clip_state
    )
    next_state = state + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
    next_state = jnp.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)
    return _clip_state(next_state) if clip_state else next_state
