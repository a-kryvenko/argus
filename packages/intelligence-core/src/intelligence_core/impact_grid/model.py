from __future__ import annotations

import numpy as np


class GridImpactModel:
    def predict(self, features: np.ndarray) -> tuple[float, float]:
        raw = 0.06 * max(-features[0], 0.0) + 0.002 * features[2] + 0.08 * features[3] + 0.03 * features[4]
        severity = float(np.tanh(max(raw, 0.0)))
        confidence = float(min(0.95, 0.55 + 0.25 * severity))
        return severity, confidence
