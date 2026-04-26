from __future__ import annotations

from datetime import datetime, timedelta, timezone
import torch

from forecast_core.inference.postprocess import sigmoid


class ForecastInferenceService:
    def __init__(self, model, config: dict, target_scaler):
        self.model = model
        self.config = config
        self.target_scaler = target_scaler
        self.regime_names = config['inference'].get('regime_names', ['quiet', 'recurrent', 'transient', 'mixed'])

    def predict(self, batch, reference_timestamp: datetime | None = None) -> dict:
        steps = self.config['model']['output_steps']
        base_ts = reference_timestamp or datetime.now(timezone.utc)
        if not batch:
            return {
                'points': [
                    {
                        'timestamp': base_ts + timedelta(hours=i + 1),
                        'bx': 0.0,
                        'by': 0.0,
                        'bz': 0.0,
                        'density': 0.0,
                        'std_bx': 1.0,
                        'std_by': 1.0,
                        'std_bz': 1.0,
                        'std_density': 1.0,
                        'southward_bz_probability': 0.5,
                    }
                    for i in range(steps)
                ],
                'regime_probabilities': {name: 1.0 / len(self.regime_names) for name in self.regime_names},
            }

        outputs = self.model(
            batch['image_embed'],
            batch['history'],
            batch['context'],
            batch['persistence'],
            batch.get('image_embed_mask'),
        )
        mean = self.target_scaler.denormalize(outputs['mean'])[0]
        std = outputs['std'][0] * self.target_scaler.std.to(outputs['std'].device)
        event_probs = sigmoid(outputs['bz_event_logits'][0])
        regime_probs = torch.softmax(outputs['regime_logits'][0], dim=-1)

        points = []
        for idx in range(mean.shape[0]):
            points.append({
                'timestamp': base_ts + timedelta(hours=idx + 1),
                'bx': float(mean[idx, 0]),
                'by': float(mean[idx, 1]),
                'bz': float(mean[idx, 2]),
                'density': float(mean[idx, 3]),
                'std_bx': float(std[idx, 0]),
                'std_by': float(std[idx, 1]),
                'std_bz': float(std[idx, 2]),
                'std_density': float(std[idx, 3]),
                'southward_bz_probability': float(event_probs[idx]),
            })
        return {
            'points': points,
            'regime_probabilities': {name: float(prob) for name, prob in zip(self.regime_names, regime_probs, strict=False)},
        }
