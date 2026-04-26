from __future__ import annotations

from pathlib import Path
import torch


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    return torch.load(Path(path), map_location=device, weights_only=False)
