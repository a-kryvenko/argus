import numpy as np
import pandas as pd
import pytest
from forecast.features import build_features


def test_hourly_features_keep_the_operational_formulas():
    frame = pd.DataFrame({
        'bx': 3., 'by': 0., 'bz': -4., 'v': np.arange(400., 408.),
        'n': 5., 't': 100000., 'kp': 2., 'ap': 5., 'dst': -10., 'f10_7': 120.,
    })
    result = build_features(frame)
    assert result.bt.eq(5.).all()
    assert result.southward_bz.eq(4.).all()
    assert result.bz_over_bt.eq(-0.8).all()
    assert result.dynamic_pressure.iloc[-1] == 5 * 407**2
    assert result.v_mean_3h.iloc[-1] == 406
    assert result.v_delta_3h.iloc[-1] == 1
    assert result.v_mean_6h.iloc[-1] == 404.5
    assert result.v_mean_7d.iloc[-1] == 403.5
    with pytest.raises(Exception, match='greater than 6'):
        build_features(frame.iloc[:6].copy())
