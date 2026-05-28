"""Vehicle dynamics: kinematic bicycle or learned SINDy (switch via :func:`get_dynamics`)."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Literal

from sindympc.dynamics import bicycle_model, sindy_model
from sindympc.dynamics.sindy_model import DEFAULT_MODEL_PATH

DynamicsName = Literal["bicycle", "sindy"]

__all__ = [
    "DynamicsName",
    "DEFAULT_MODEL_PATH",
    "bicycle_model",
    "get_dynamics",
    "sindy_model",
]


def get_dynamics(
    name: DynamicsName = "bicycle",
    *,
    sindy_model_path: Path | str | None = None,
) -> ModuleType:
    """Return a dynamics module with ``continuous_dynamics`` and ``rk4_step``.

    RRT planning is geometric and does not call this; use the same ``name`` in
    ``scripts/03_run_mpc_sim.py`` so MPC rollouts match the simulated plant.

    Args:
        name: ``"bicycle"`` (kinematic model) or ``"sindy"`` (``data/sindy_model.npz``).
        sindy_model_path: Override path to ``.npz`` (or legacy ``.pkl``) when ``name="sindy"``.

    Returns:
        Either :mod:`bicycle_model` or :mod:`sindy_model` (after loading coefficients).
    """
    if name == "bicycle":
        return bicycle_model
    if name == "sindy":
        path = Path(sindy_model_path) if sindy_model_path is not None else DEFAULT_MODEL_PATH
        sindy_model.load_model(path)
        return sindy_model
    raise ValueError(f"Unknown dynamics {name!r}. Choose 'bicycle' or 'sindy'.")
