"""Density DLinear blended with frozen speed-assisted LightGBM heads, no neural runtime."""
import logging
import numpy as np
import pandas as pd
from common.data.omni import OMNI_FILL_VALUES
from forecast.inference.density_dlinear import DensityDLinearForecaster
from forecast.inference.proswin_blend import select_predictions

FORMAT = 'density_proswin_blend'
VERSION = 'argus-plasma-density-proswin-blend-v1'
FEATURES = [f'{target}_{name}' for target in ('n', 'v') for name in
            ('speed', 'age_hours', 'std_24h', 'count_24h', 'change_3h', 'change_6h', 'change_12h', 'change_24h')] + ['proswin']
logger = logging.getLogger(__name__)


def history_features(issue, history, variable):
    """Training-equivalent features from unfilled availability-stamped hourly values."""
    h = history[['issue_time', variable]].copy()
    h['issue_time'] = pd.to_datetime(h.issue_time, utc=True)
    h = h[np.isfinite(h[variable]) & h[variable].ge(0) & h[variable].ne(OMNI_FILL_VALUES[variable])]
    h = h.sort_values('issue_time')
    if h.groupby('issue_time')[variable].nunique().gt(1).any():
        raise ValueError('Conflicting hourly history')
    h = h.drop_duplicates('issue_time')
    times = pd.DatetimeIndex(h.issue_time).asi8
    values = h[variable].to_numpy(float)
    hour = pd.Timedelta(hours=1).value
    t = issue.value
    end = np.searchsorted(times, t, side='right')
    start = np.searchsorted(times, t-24*hour, side='right')
    age = (t-times[end-1])/hour if end else np.nan
    row = {'speed': values[end-1] if end and age <= 2 else np.nan, 'age_hours': age,
           'std_24h': float(np.std(values[start:end])) if end-start >= 12 else np.nan,
           'count_24h': end-start}
    for hours in (3, 6, 12, 24):
        pos = np.searchsorted(times, t-hours*hour, side='right')-1
        recent = pos >= 0 and (t-hours*hour-times[pos])/hour <= 2
        row[f'change_{hours}h'] = row['speed']-values[pos] if recent else np.nan
    return {f'{variable}_{key}': value for key, value in row.items()}


class DensityProswinForecaster:
    def __init__(self, bundle):
        if bundle.get('format') != FORMAT or bundle.get('version') != 1:
            raise ValueError('Unsupported density blend artifact')
        self.dlinear = DensityDLinearForecaster(bundle['dlinear'])
        self.weights = np.asarray(bundle['blend_weights'], float)
        self.offsets = np.asarray(bundle['interval_offsets'], float)
        self.features = bundle['features']
        if (self.dlinear.horizon < 96 or self.features != FEATURES or
                self.weights.shape != (96,) or self.offsets.shape != (96, 2) or
                not np.isfinite(self.weights).all() or not np.isfinite(self.offsets).all() or
                ((self.weights < 0) | (self.weights > 1)).any() or
                (self.offsets[:, 0] > self.offsets[:, 1]).any() or len(bundle['heads']) != 96):
            raise ValueError('Invalid density blend configuration')
        import lightgbm as lgb
        self._prediction_error = lgb.basic.LightGBMError
        self.heads = [lgb.Booster(model_str=s) for s in bundle['heads']]
        if any(head.feature_name() != FEATURES for head in self.heads):
            raise ValueError('Density head feature order mismatch')
        self.last_status = {}
        self.allowed_model_versions = ('proswin-fold1-nrt-v1',)

    def frame(self, issue_time, density_history, speed_history, predictions=(), *, ready_at=None, source_cutoff=None):
        base = self.dlinear.frame(issue_time, density_history)
        issue = pd.Timestamp(base.issue_time.iloc[0])
        p = select_predictions(issue, base.valid_time, predictions, ready_at=ready_at, source_cutoff=source_cutoff,
                               allowed_model_versions=self.allowed_model_versions)
        row = {**history_features(issue, density_history, 'n'), **history_features(issue, speed_history, 'v')}
        x = pd.DataFrame([row] * len(base)); x['proswin'] = p
        x = x[self.features].astype(np.float32)
        active = np.isfinite(x[['n_speed', 'v_speed', 'proswin']]).all(axis=1).to_numpy()
        active &= np.arange(len(base)) < 96
        used = np.zeros(len(base), dtype=bool)
        for i in np.flatnonzero(active):
            try:
                direct = float(self.heads[i].predict(x.iloc[[i]], num_threads=1)[0])
                if not np.isfinite(direct):
                    raise ValueError('Nonfinite density prediction')
                point = (1-self.weights[i])*base.loc[i, 'n_q50'] + self.weights[i]*max(direct, 0)
                low, high = self.offsets[i]
                base.loc[i, ['n_q10', 'n_q50', 'n_q90']] = [max(min(point+low, point), 0), point, max(point+high, point)]
                used[i] = True
            except (ValueError, RuntimeError, self._prediction_error):
                logger.exception('Density candidate failed for lead %s; retaining DLinear', i+1)
        self.last_status = {'candidate_leads': int(used.sum()), 'fallback_leads': int((~used).sum())}
        base['proswin_weight'] = 0.
        base.loc[used, 'proswin_weight'] = self.weights[np.flatnonzero(used)]
        return base
