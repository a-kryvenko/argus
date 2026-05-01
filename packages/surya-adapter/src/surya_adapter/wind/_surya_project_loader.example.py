"""Project-local Surya glue template.

Copy this file to `surya_adapter/_surya_project_loader.py` and wire it to your
vendored NASA Surya checkout. This file is intentionally not imported by
default because every Surya checkout/checkpoint layout may differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_solar_wind_model(device: torch.device) -> torch.nn.Module:
    """Return a loaded pretrained Surya solar-wind model.

    Expected by surya_adapter.solar_wind._SuryaSolarWindBackend.
    Replace this with the exact loader from your vendored Surya downstream
    example/checkpoint code.
    """
    raise NotImplementedError("Connect this function to vendor/surya solar-wind loader")


def build_solar_wind_batch(
    *,
    observation: Any,
    aia_paths: dict[str, Path],
    hmi_paths: dict[str, Path],
    lead_hour: int,
) -> dict[str, torch.Tensor]:
    """Build one batch for Surya solar-wind inference.

    Use Surya's own preprocessing/scalers/channel order here. The adapter will
    move tensors to the selected device and call model(batch).
    """
    raise NotImplementedError("Connect this function to Surya dataset/preprocessing code")
