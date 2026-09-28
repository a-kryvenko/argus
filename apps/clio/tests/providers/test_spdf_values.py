from unittest.mock import Mock

import numpy as np
import pandas as pd

from clio.providers.spdf_loader import SPDF_Loader


def test_cdf_fill_and_invalid_values_are_masked_before_hourly_selection():
    cdf = Mock()
    cdf.varget.return_value = np.array([-1e31, 420., np.inf, 5000.], dtype=np.float32)
    cdf.varattsget.return_value = {
        'FILLVAL': np.float32(-1e31), 'VALIDMIN': 0., 'VALIDMAX': 3000.,
    }
    values = SPDF_Loader._values(cdf, 'Vp')
    np.testing.assert_allclose(values, [np.nan, 420., np.nan, np.nan], equal_nan=True)
    series = pd.Series(values, index=pd.date_range('2026-09-28', periods=4, freq='10min'))
    assert series.resample('1h').first().iloc[0] == 420.


def test_signed_magnetic_components_and_vector_bounds():
    cdf = Mock()
    cdf.varget.return_value = [[-4., 2., -1e31], [5., np.nan, 6.]]
    cdf.varattsget.return_value = {'FILLVAL': -1e31, 'VALIDMIN': [-100., -100., -100.],
                                 'VALIDMAX': [100., 100., 100.]}
    np.testing.assert_allclose(SPDF_Loader._values(cdf, 'BGSEc'),
                               [[-4., 2., np.nan], [5., np.nan, 6.]], equal_nan=True)
