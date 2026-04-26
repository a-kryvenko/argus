from __future__ import annotations

from pathlib import Path
import torch


class BestCheckpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.best = float("inf")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, metric: float, payload: dict) -> bool:
        if metric >= self.best:
            return False
        self.best = metric
        torch.save(payload, self.path)
        return True
