from __future__ import annotations

import numpy as np


def build_satdrag_features(payload: dict) -> np.ndarray:
    density = np.asarray(payload.get("density_proxy", []), dtype=float)
    speed = np.asarray(payload.get("speed_forecast", []), dtype=float)
    bz = np.asarray(payload.get("bz_forecast", []), dtype=float)
    return np.array([
        float(np.max(density)) if density.size else 0.0,
        float(np.mean(density)) if density.size else 0.0,
        float(np.max(speed)) if speed.size else 0.0,
        float(np.mean(np.maximum(-bz, 0.0))) if bz.size else 0.0,
    ], dtype=float)
