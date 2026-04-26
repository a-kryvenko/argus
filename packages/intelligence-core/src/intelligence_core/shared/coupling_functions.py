from __future__ import annotations

import numpy as np


def dawn_dusk_electric_field(speed: np.ndarray, bz: np.ndarray) -> np.ndarray:
    return np.asarray(speed) * np.maximum(-np.asarray(bz), 0.0) * 1e-3
