"""SINDy feature libraries aligned with kinematic bicycle structure."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from pysindy.feature_library.base import BaseFeatureLibrary, x_sequence_or_item
from pysindy.utils import AxesArray

# Column order for SINDYc inputs: [x, y, theta, v, throttle, delta]
IDX_THETA = 2
IDX_V = 3
IDX_DELTA = 5
MAX_DELTA_FOR_TAN = 1.3  # stay below pi/2 singularity


def _as_2d_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _trig_block(x: np.ndarray) -> np.ndarray:
    theta = x[:, IDX_THETA]
    v = x[:, IDX_V]
    delta = x[:, IDX_DELTA]
    tan_delta = np.tan(np.clip(delta, -MAX_DELTA_FOR_TAN, MAX_DELTA_FOR_TAN))
    return np.column_stack(
        [
            np.sin(theta),
            np.cos(theta),
            tan_delta,
            v * np.sin(theta),
            v * np.cos(theta),
            v * tan_delta,
        ]
    )


class BicycleFeatureLibrary(BaseFeatureLibrary):
    """Θ(x, u) with polynomials plus bicycle-consistent trigonometric terms.

    Explicit terms (beyond degree-``poly_degree`` polynomials on all inputs):
        sin(θ), cos(θ), tan(δ), v·sin(θ), v·cos(θ), v·tan(δ)

    These mirror the kinematic bicycle structure:
        ẋ ≈ v cos(θ),  ẏ ≈ v sin(θ),  θ̇ ≈ v tan(δ) / L,  v̇ = a
    while polynomials capture BeamNG residuals and coupling.
    """

    def __init__(self, poly_degree: int = 2) -> None:
        self.poly_degree = poly_degree
        # Use only local/pose + inputs for polynomial terms (translation-invariant).
        # Full input ordering is [x, y, theta, v, throttle, delta].
        self.poly_input_indices = np.array([2, 3, 4, 5], dtype=np.int32)
        self._poly = PolynomialFeatures(
            degree=poly_degree,
            include_bias=False,
        )
        self.n_trig_features = 6
        self.powers_: Optional[np.ndarray] = None

    @x_sequence_or_item
    def fit(self, x_full, y: None = None) -> "BicycleFeatureLibrary":
        x_stack = np.vstack([_as_2d_array(x) for x in x_full])
        self.n_features_in_ = x_stack.shape[1]
        self._poly.fit(x_stack[:, self.poly_input_indices])
        self.powers_ = np.asarray(self._poly.powers_, dtype=np.int32)
        n_poly = self._poly.transform(x_stack[:1, self.poly_input_indices]).shape[1]
        self.n_output_features_ = n_poly + self.n_trig_features
        return self

    @x_sequence_or_item
    def transform(self, x_full) -> list:
        outputs = []
        for x in x_full:
            arr = _as_2d_array(x)
            poly_in = arr[:, self.poly_input_indices]
            features = np.hstack([self._poly.transform(poly_in), _trig_block(arr)])
            outputs.append(AxesArray(features, {"ax_time": 0, "ax_coord": 1}))
        return outputs

    def get_feature_names(
        self, input_features: Optional[Sequence[str]] = None
    ) -> List[str]:
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        poly_features = [input_features[i] for i in self.poly_input_indices.tolist()]
        poly_names = self._poly.get_feature_names_out(poly_features).tolist()
        trig_names = [
            "sin(theta)",
            "cos(theta)",
            "tan(delta)",
            "v sin(theta)",
            "v cos(theta)",
            "v tan(delta)",
        ]
        return poly_names + trig_names
