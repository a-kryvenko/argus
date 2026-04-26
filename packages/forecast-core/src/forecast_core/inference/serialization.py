from __future__ import annotations

from pathlib import Path
import torch


def load_checkpoint(model, path: str | Path):
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    model.load_state_dict(state)
    return payload
