"""SINDy feature libraries aligned with kinematic bicycle structure."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from pysindy.feature_library.base import BaseFeatureLibrary, x_sequence_or_item
from pysindy.utils import AxesArray

LibraryMode = Literal["kinematic", "residual"]

# Column order for SINDYc inputs: [x, y, theta, v, u0, delta]
# kinematic: u0 = acceleration a [m/s^2]; residual: u0 = throttle [0, 1]
IDX_THETA = 2
IDX_V = 3
IDX_U0 = 4
IDX_DELTA = 5
MAX_DELTA_FOR_TAN = 1.3

# Kinematic library (matches bicycle_model + MPC acceleration)
KINEMATIC_FEATURE_NAMES: tuple[str, ...] = (
    "v cos(theta)",
    "v sin(theta)",
    "v tan(delta) / L",
    "a",
)


def _as_2d_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


class BicycleFeatureLibrary(BaseFeatureLibrary):
    """Θ(x, u) for vehicle SINDYc.

    **kinematic** — minimal bicycle-consistent basis (compare to ``bicycle_model``):
        ẋ = v cos θ,  ẏ = v sin θ,  θ̇ = (v tan δ)/L,  v̇ = a

    **residual** — kinematic block (throttle from logs) + small polynomials + extras
    for BeamNG / unmodeled physics (useful under noise or in sim).
    """

    def __init__(
        self,
        mode: LibraryMode = "residual",
        poly_degree: int = 2,
        wheel_base: float = 2.5,
    ) -> None:
        self.mode: LibraryMode = mode
        self.wheel_base = float(wheel_base)
        self.poly_degree = 0 if mode == "kinematic" else poly_degree
        self.poly_input_indices = np.array([2, 3, 4, 5], dtype=np.int32)
        self._poly = PolynomialFeatures(
            degree=max(1, self.poly_degree) if self.poly_degree > 0 else 1,
            include_bias=False,
        )
        self.n_kinematic_features = len(KINEMATIC_FEATURE_NAMES)
        self.n_residual_trig_features = 0 if mode == "kinematic" else 3
        self.powers_: Optional[np.ndarray] = None
        self._poly_active = self.poly_degree > 0
        self._use_acceleration = mode == "kinematic"

    def _kinematic_block(self, x: np.ndarray) -> np.ndarray:
        theta = x[:, IDX_THETA]
        v = x[:, IDX_V]
        delta = x[:, IDX_DELTA]
        u0 = x[:, IDX_U0]
        tan_delta = np.tan(np.clip(delta, -MAX_DELTA_FOR_TAN, MAX_DELTA_FOR_TAN))
        yaw_rate_feat = v * tan_delta / self.wheel_base
        return np.column_stack(
            [
                v * np.cos(theta),
                v * np.sin(theta),
                yaw_rate_feat,
                u0,
            ]
        )

    def _residual_kinematic_block(self, x: np.ndarray) -> np.ndarray:
        """Legacy residual core: throttle instead of ``a`` (log-aligned)."""
        theta = x[:, IDX_THETA]
        v = x[:, IDX_V]
        delta = x[:, IDX_DELTA]
        throttle = x[:, IDX_U0]
        tan_delta = np.tan(np.clip(delta, -MAX_DELTA_FOR_TAN, MAX_DELTA_FOR_TAN))
        return np.column_stack(
            [
                v * np.cos(theta),
                v * np.sin(theta),
                v * tan_delta,
                throttle,
            ]
        )

    @staticmethod
    def _residual_trig_block(x: np.ndarray) -> np.ndarray:
        theta = x[:, IDX_THETA]
        delta = x[:, IDX_DELTA]
        tan_delta = np.tan(np.clip(delta, -MAX_DELTA_FOR_TAN, MAX_DELTA_FOR_TAN))
        return np.column_stack(
            [np.sin(theta), np.cos(theta), tan_delta],
        )

    @x_sequence_or_item
    def fit(self, x_full, y: None = None) -> "BicycleFeatureLibrary":
        x_stack = np.vstack([_as_2d_array(x) for x in x_full])
        self.n_features_in_ = x_stack.shape[1]

        if self._poly_active:
            self._poly.fit(x_stack[:, self.poly_input_indices])
            self.powers_ = np.asarray(self._poly.powers_, dtype=np.int32)
            n_poly = self._poly.transform(x_stack[:1, self.poly_input_indices]).shape[1]
        else:
            self.powers_ = np.zeros((0, len(self.poly_input_indices)), dtype=np.int32)
            n_poly = 0

        self.n_output_features_ = (
            n_poly + self.n_kinematic_features + self.n_residual_trig_features
        )
        return self

    @x_sequence_or_item
    def transform(self, x_full) -> list:
        outputs = []
        kin_fn = (
            self._kinematic_block
            if self._use_acceleration
            else self._residual_kinematic_block
        )
        for x in x_full:
            arr = _as_2d_array(x)
            blocks = [kin_fn(arr)]
            if self._poly_active:
                poly_in = arr[:, self.poly_input_indices]
                blocks.insert(0, self._poly.transform(poly_in))
            if self.n_residual_trig_features > 0:
                blocks.append(self._residual_trig_block(arr))
            features = np.hstack(blocks)
            outputs.append(AxesArray(features, {"ax_time": 0, "ax_coord": 1}))
        return outputs

    def get_feature_names(
        self, input_features: Optional[Sequence[str]] = None
    ) -> List[str]:
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        names: List[str] = []
        if self._poly_active:
            poly_features = [
                input_features[i] for i in self.poly_input_indices.tolist()
            ]
            names.extend(self._poly.get_feature_names_out(poly_features).tolist())
        if self._use_acceleration:
            names.extend(KINEMATIC_FEATURE_NAMES)
        else:
            names.extend(
                [
                    "v cos(theta)",
                    "v sin(theta)",
                    "v tan(delta)",
                    "throttle",
                ]
            )
        if self.n_residual_trig_features > 0:
            names.extend(["sin(theta)", "cos(theta)", "tan(delta)"])
        return names
