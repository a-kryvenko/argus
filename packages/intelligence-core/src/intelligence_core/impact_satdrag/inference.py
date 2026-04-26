from __future__ import annotations

from intelligence_core.impact_satdrag.features import build_satdrag_features
from intelligence_core.impact_satdrag.model import SatDragModel


class SatDragInferenceService:
    def __init__(self) -> None:
        self.model = SatDragModel()

    def predict(self, payload: dict) -> dict:
        risk, uncertainty = self.model.predict(build_satdrag_features(payload))
        confidence = min(0.95, 0.5 + 0.3 * risk)
        return {"drag_risk_score": risk, "uncertainty_inflation": uncertainty, "confidence": confidence}
