from __future__ import annotations

import torch
import torch.nn as nn

from forecast_core.fusion_decoder.multimodal_decoder import MultimodalDecoder
from forecast_core.probabilistic_heads.event_head import SouthwardBzEventHead
from forecast_core.probabilistic_heads.regression_head import ProbabilisticRegressionHead
from forecast_core.regime_classifier.regime_head import RegimeClassifier
from forecast_core.solar_encoder.surya_adapter import SuryaAdapter, SuryaAdapterConfig
from forecast_core.timeseries_encoder.history_encoder import HistoryEncoder


class ForecastModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        mcfg = config['model']
        self.output_steps = mcfg['output_steps']
        self.output_features = mcfg['output_features']
        self.solar = SuryaAdapter(SuryaAdapterConfig(
            embed_dim=mcfg['embed_dim'],
            projection_dim=mcfg['solar_proj_dim'],
            dropout=mcfg.get('dropout', 0.1),
        ))
        self.history = HistoryEncoder(
            input_dim=mcfg['history_input_dim'],
            hidden_dim=mcfg['history_hidden_dim'],
            output_dim=mcfg['history_proj_dim'],
            dropout=mcfg.get('dropout', 0.1),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(mcfg['context_input_dim']),
            nn.Linear(mcfg['context_input_dim'], mcfg['context_proj_dim']),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(mcfg['solar_proj_dim'] + mcfg['history_proj_dim'] + mcfg['context_proj_dim'], mcfg['fusion_dim']),
            nn.GELU(),
            nn.Dropout(mcfg.get('dropout', 0.1)),
        )
        self.decoder = MultimodalDecoder(
            model_dim=mcfg['fusion_dim'],
            lead_dim=mcfg['lead_dim'],
            num_heads=mcfg['num_heads'],
            depth=mcfg['depth'],
            mlp_ratio=mcfg['mlp_ratio'],
            dropout=mcfg.get('dropout', 0.1),
            output_steps=self.output_steps,
        )
        decoder_dim = mcfg['fusion_dim'] + mcfg['lead_dim']
        self.regression_head = ProbabilisticRegressionHead(decoder_dim, mcfg['head_hidden_dim'], self.output_features)
        self.event_head = SouthwardBzEventHead(decoder_dim, mcfg['head_hidden_dim'])
        self.regime_classifier = RegimeClassifier(mcfg['fusion_dim'], mcfg['head_hidden_dim'], mcfg.get('num_regimes', 4))

    def forward(
        self,
        image_embed: torch.Tensor,
        history: torch.Tensor,
        context: torch.Tensor,
        persistence: torch.Tensor,
        image_embed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        solar_emb = self.solar(image_embed, image_embed_mask)
        hist_emb = self.history(history)
        ctx_emb = self.context_proj(context)
        fused = self.fusion(torch.cat([solar_emb, hist_emb, ctx_emb], dim=-1))
        decoded = self.decoder(fused)
        persistence = persistence.unsqueeze(1).expand(-1, decoded.shape[1], -1)
        mean, std = self.regression_head(decoded, persistence)
        bz_event_logits = self.event_head(decoded)
        regime_logits = self.regime_classifier(fused)
        return {
            'mean': mean,
            'std': std,
            'bz_event_logits': bz_event_logits,
            'regime_logits': regime_logits,
        }
