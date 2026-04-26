from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch


def load_vendor_object(import_path: str):
    module_path, _, attr = import_path.partition(":")
    if not module_path or not attr:
        raise ValueError("import_path must look like 'package.module:object'")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload does not contain a usable state dict")
    cleaned = {}
    for k, v in payload.items():
        if isinstance(k, str) and k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def load_vendor_model(
    factory: str,
    init_kwargs: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> torch.nn.Module:
    init_kwargs = init_kwargs or {}
    ctor = load_vendor_object(factory)
    model = ctor(**init_kwargs)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dict = _extract_state_dict(payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if strict is False:
            if missing:
                print(f"[surya-adapter] missing keys while loading {checkpoint_path}: {missing[:10]}")
            if unexpected:
                print(f"[surya-adapter] unexpected keys while loading {checkpoint_path}: {unexpected[:10]}")
    return model
