from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml
import torch


@dataclass(slots=True)
class TensorScalerState:
    mean: list[float]
    std: list[float]


class TensorScaler:
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-6)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class ForecastScalers:
    def __init__(self, history: TensorScaler, context: TensorScaler, target: TensorScaler):
        self.history = history
        self.context = context
        self.target = target

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ForecastScalers":
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(
            history=TensorScaler(cfg["history_mean"], cfg["history_std"]),
            context=TensorScaler(cfg["context_mean"], cfg["context_std"]),
            target=TensorScaler(cfg["target_mean"], cfg["target_std"]),
        )


def save_scalers(path: str | Path, payload: dict) -> None:
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
