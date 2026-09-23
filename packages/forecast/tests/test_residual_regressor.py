import io

import joblib
import numpy as np
import pandas as pd
import pytest

from forecast.inference.residual import ResidualRegressor


class Correction:
    def predict(self, frame):
        return np.array([-20., 40.])


def test_residual_restores_speed_after_serialization():
    model = ResidualRegressor(Correction(), correction_scale=0.5)
    stream = io.BytesIO()
    joblib.dump(model, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    np.testing.assert_allclose(restored.predict(pd.DataFrame({'dlinear_v': [400., 500.]})), [390., 520.])


def test_zero_scale_is_exact_baseline_without_calling_model():
    model = ResidualRegressor(None, correction_scale=0)
    np.testing.assert_array_equal(model.predict(pd.DataFrame({'dlinear_v': [400., 500.]})), [400., 500.])
    with pytest.raises(ValueError, match='finite baseline'):
        model.predict(pd.DataFrame({'dlinear_v': [np.nan]}))


@pytest.mark.parametrize('scale', [-1, 2, np.nan])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match='correction_scale'):
        ResidualRegressor(None, correction_scale=scale)
