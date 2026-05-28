"""PySINDy training for vehicle dynamics (library module — no script entry point)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pysindy as ps
from pysindy.differentiation import SINDyDerivative

from sindympc.system_id.sindy_features import BicycleFeatureLibrary

# State / control column semantics for BeamNG telemetry
STATE_NAMES: Tuple[str, ...] = ("x", "y", "theta", "v")
N_STATES = len(STATE_NAMES)

CONTROL_NAMES: Tuple[str, ...] = ("a", "delta")
N_CONTROLS = len(CONTROL_NAMES)

# Savitzky–Golay: 11-sample window at dt=0.1 s
SG_WINDOW_LENGTH = 11
SG_POLYORDER = 3

# STLSQ sparsity (raise slightly — richer library than pure cubic polynomials)
STLSQ_THRESHOLD = 0.02
STLSQ_ALPHA = 0.05

# Low-order polynomials on [x, y, θ, v, a, δ]; sin/cos/tan terms handle heading & steering
POLYNOMIAL_DEGREE = 2


def build_feature_library() -> BicycleFeatureLibrary:
    """Candidate functions Θ(x, u) with bicycle-consistent trigonometric terms."""
    return BicycleFeatureLibrary(poly_degree=POLYNOMIAL_DEGREE)


def train_sindy_model(
    states: np.ndarray,
    controls: np.ndarray,
    dt: float,
) -> ps.SINDy:
    """Fit sparse SINDy (SINDYc) on BeamNG-aligned ``[x, y, θ, v]`` with ``[a, δ]``.

    Feature library combines:
        - Polynomials (degree 2) on all inputs for residual nonlinearities
        - sin(θ), cos(θ), tan(δ), v·sin(θ), v·cos(θ), v·tan(δ) for bicycle structure

    Args:
        states: Shape ``(n_samples, 4)``.
        controls: Shape ``(n_samples, 2)`` — throttle ``a``, steering ``delta``.
        dt: Sample period [s].

    Returns:
        Fitted :class:`pysindy.SINDy` model.
    """
    states = np.asarray(states, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != N_STATES:
        raise ValueError(f"Expected states shape (n, {N_STATES}), got {states.shape}")
    if controls.ndim != 2 or controls.shape[1] != N_CONTROLS:
        raise ValueError(
            f"Expected controls shape (n, {N_CONTROLS}), got {controls.shape}"
        )
    if len(states) != len(controls):
        raise ValueError("states and controls must have the same number of rows.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    sg_half_width = (SG_WINDOW_LENGTH - 1) // 2
    differentiation_method = SINDyDerivative(
        kind="savitzky_golay",
        order=SG_POLYORDER,
        left=sg_half_width,
        right=sg_half_width,
        iwindow=True,
    )

    feature_library = build_feature_library()
    optimizer = ps.STLSQ(threshold=STLSQ_THRESHOLD, alpha=STLSQ_ALPHA)

    model = ps.SINDy(
        differentiation_method=differentiation_method,
        feature_library=feature_library,
        optimizer=optimizer,
    )

    model.fit(
        x=states,
        u=controls,
        t=dt,
        feature_names=list(STATE_NAMES) + list(CONTROL_NAMES),
    )

    return model
