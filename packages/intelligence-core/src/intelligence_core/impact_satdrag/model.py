from __future__ import annotations

import numpy as np


class SatDragModel:
    def predict(self, features: np.ndarray) -> tuple[float, float]:
        raw = 0.15 * features[0] + 0.08 * features[1] + 0.001 * features[2] + 0.04 * features[3]
        risk = float(np.tanh(max(raw, 0.0)))
        uncertainty = float(1.0 + 1.5 * risk)
        return risk, uncertainty
