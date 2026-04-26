from __future__ import annotations

import torch
import torch.nn as nn


class MultimodalDecoder(nn.Module):
    def __init__(self, model_dim: int, lead_dim: int, num_heads: int, depth: int, mlp_ratio: float, dropout: float, output_steps: int):
        super().__init__()
        self.output_steps = output_steps
        self.lead_emb = nn.Embedding(output_steps, lead_dim)
        self.fused_dim = model_dim + lead_dim
        self.pos = nn.Parameter(torch.randn(1, output_steps, self.fused_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fused_dim,
            nhead=num_heads,
            dim_feedforward=int(self.fused_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, fused_context: torch.Tensor) -> torch.Tensor:
        bsz = fused_context.shape[0]
        lead_idx = torch.arange(self.output_steps, device=fused_context.device)
        lead_emb = self.lead_emb(lead_idx)[None, :, :].expand(bsz, -1, -1)
        x = torch.cat([fused_context[:, None, :].expand(-1, self.output_steps, -1), lead_emb], dim=-1)
        x = x + self.pos
        return self.encoder(x)
