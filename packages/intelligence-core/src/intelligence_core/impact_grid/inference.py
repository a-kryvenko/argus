from __future__ import annotations

from intelligence_core.impact_grid.features import build_grid_features
from intelligence_core.impact_grid.model import GridImpactModel


class GridImpactInferenceService:
    def __init__(self) -> None:
        self.model = GridImpactModel()

    def predict(self, payload: dict) -> dict:
        severity, confidence = self.model.predict(build_grid_features(payload))
        if severity < 0.25:
            level = "low"
        elif severity < 0.5:
            level = "guarded"
        elif severity < 0.75:
            level = "elevated"
        else:
            level = "high"
        return {"severity_score": severity, "risk_level": level, "confidence": confidence}
