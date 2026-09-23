"""Serializable baseline-plus-residual regressor used by quantile bundles."""
import numpy as np


class ResidualRegressor:
    """Restore absolute predictions from a fitted residual model."""

    def __init__(self, model, baseline_column="dlinear_v", correction_scale=1.0):
        if not np.isfinite(correction_scale) or not 0 <= correction_scale <= 1:
            raise ValueError("correction_scale must be between zero and one")
        self.model = model
        self.baseline_column = baseline_column
        self.correction_scale = float(correction_scale)

    def predict(self, frame):
        baseline = frame[self.baseline_column].to_numpy(dtype=float)
        if not np.isfinite(baseline).all():
            raise ValueError("Residual predictions require finite baseline values")
        if self.correction_scale == 0:
            return baseline
        return baseline + self.correction_scale * self.model.predict(frame)
