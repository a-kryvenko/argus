"""History-only density DLinear with frozen residual quantiles, without PyTorch."""
import numpy as np
import pandas as pd

from common.data.omni import OMNI_FILL_VALUES
from forecast.inference.rotation_dlinear import RotationDLinearForecaster


class DensityDLinearForecaster(RotationDLinearForecaster):
    registry_name = 'plasma_density_quantile'
    artifact_format = 'density_dlinear'
    variable = 'n'
    history_name = 'density'

    def __init__(self, bundle, *, batch_size=256):
        super().__init__(bundle, batch_size=batch_size)
        if self.settings.get('scale') != 'original':
            raise ValueError('Density DLinear requires original-scale normalization')
        if (self.settings.get('point_postprocessing') != 'maximum(point, 0)'
                or self.settings.get('quantile_postprocessing') != 'maximum(point + residual_quantile, 0)'):
            raise ValueError('Unsupported density DLinear postprocessing')
        if not np.array_equal(bundle.get('quantiles'), [.1, .5, .9]):
            raise ValueError('Density DLinear requires q10/q50/q90')
        self.residual_offsets = np.array(bundle['residual_offsets'], dtype=float, copy=True)
        if (self.residual_offsets.shape != (self.horizon, 3)
                or not np.isfinite(self.residual_offsets).all()
                or (np.diff(self.residual_offsets, axis=1) < 0).any()):
            raise ValueError('Invalid density residual quantiles')

    def _clean_values(self, values):
        # Match training: remove fill code, negatives and nonfinite values;
        # do not trim genuine high-density events.
        return values.where(np.isfinite(values) & values.ge(0) & values.ne(OMNI_FILL_VALUES['n']))

    def frame(self, issue_time, density_history):
        issue = pd.Timestamp(issue_time)
        if issue.tzinfo is None:
            raise ValueError('Timezone-aware issue_time required')
        issue = issue.tz_convert('UTC').floor('h')
        leads = np.arange(1, self.horizon + 1)
        request = pd.DataFrame({'issue_time': issue, 'lead_hours': leads})
        result = self._add_predictions(request, density_history, column='dlinear_n', require_history=True)
        point = np.maximum(result.dlinear_n.to_numpy(), 0)
        quantiles = np.maximum(point[:, None] + self.residual_offsets, 0)
        if not np.isfinite(quantiles).all():
            raise ValueError('Nonfinite density quantiles')
        result['valid_time'] = issue + pd.to_timedelta(leads, unit='h')
        result['dlinear_n'] = point
        result[['n_q10', 'n_q50', 'n_q90']] = quantiles
        return result
