from __future__ import annotations

import torch
import torch.nn.functional as F

from forecast_core.probabilistic_heads.uncertainty import gaussian_nll


def compute_forecast_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, bz_event_target: torch.Tensor, loss_weights: torch.Tensor, time_weights: torch.Tensor, southward_weight: float = 2.0) -> tuple[torch.Tensor, dict[str, float]]:
    mean = outputs["mean"]
    std = outputs["std"]
    bz_event_logits = outputs["bz_event_logits"]

    reg_loss = gaussian_nll(mean, std, target)
    southward_mask = (target[..., 2] < 0).float() * (southward_weight - 1.0) + 1.0
    reg_loss[..., 2] = reg_loss[..., 2] * southward_mask
    reg_loss = reg_loss * loss_weights.view(1, 1, -1)
    reg_loss = reg_loss.mean(dim=-1) * time_weights.view(1, -1)
    reg_loss = reg_loss.mean()

    event_loss = F.binary_cross_entropy_with_logits(bz_event_logits, bz_event_target)
    total = reg_loss + 0.3 * event_loss
    return total, {"regression_loss": float(reg_loss.detach().cpu()), "event_loss": float(event_loss.detach().cpu())}
