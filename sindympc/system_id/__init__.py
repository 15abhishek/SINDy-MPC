"""System identification (PySINDy) for vehicle dynamics."""

from sindympc.system_id.sindy_training import (
    CONTROL_NAMES,
    N_CONTROLS,
    N_STATES,
    STATE_NAMES,
    train_sindy_model,
)

__all__ = [
    "CONTROL_NAMES",
    "N_CONTROLS",
    "N_STATES",
    "STATE_NAMES",
    "train_sindy_model",
]
