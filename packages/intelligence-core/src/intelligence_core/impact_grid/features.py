from __future__ import annotations

import numpy as np
from intelligence_core.shared.coupling_functions import dawn_dusk_electric_field


def build_grid_features(payload: dict) -> np.ndarray:
    bz = np.asarray(payload.get("bz_forecast", []), dtype=float)
    speed = np.asarray(payload.get("speed_forecast", []), dtype=float)
    kp = np.asarray(payload.get("kp_forecast", []), dtype=float)
    ey = dawn_dusk_electric_field(speed if speed.size else np.zeros_like(bz), bz if bz.size else np.zeros_like(speed))
    return np.array([
        float(np.min(bz)) if bz.size else 0.0,
        float(np.mean(np.maximum(-bz, 0.0))) if bz.size else 0.0,
        float(np.max(speed)) if speed.size else 0.0,
        float(np.max(kp)) if kp.size else 0.0,
        float(np.max(ey)) if ey.size else 0.0,
    ], dtype=float)
