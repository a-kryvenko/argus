from __future__ import annotations

import torch


def gaussian_nll(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = std.square() + 1e-6
    return 0.5 * (((target - mean).square() / var) + torch.log(var))
