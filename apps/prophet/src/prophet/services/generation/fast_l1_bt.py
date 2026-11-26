"""Short probabilities for hourly mean of measured minute Bt after an L1 front."""
from hashlib import sha256

import numpy as np
import pandas as pd

from prophet.services.generation.fast_l1_model import (
    HOUR, MAX_TRIGGER_AGE, MINUTE, PRESSURE_FACTOR, _utc_index, _utc_issue,
)
from prophet.services.inputs.l1_geomagnetic import STANDALONE_WIND


THRESHOLDS = (5, 10, 15)
WINDOWS = (5, 30, 60)
FEATURES = tuple(sorted((
    *(f'x_{metric}_mean_{width}m' for metric in ('bt', 'bs', 'bz') for width in WINDOWS),
    'x_bt_max_30m', 'x_bs_max_30m', 'x_bs10_fraction_30m',
    'x_bt_baseline', 'x_bt_jump', 'x_speed_5m', 'x_density_5m', 'x_pressure_log_ratio', 'x_lead_hours',
)))


def wind_features(wind, trigger, issue):
    clock = _utc_index(wind.index, MINUTE)
    cutoff = trigger.trigger_time
    history = wind.set_axis(clock).loc[cutoff - 59 * MINUTE:cutoff]
    values = {}
    spacecraft = history.at[cutoff, 'spacecraft'] if 'spacecraft' in history and cutoff in history.index else None
    for metric, (field, receipt) in STANDALONE_WIND.items():
        series = history[field]
        if receipt in history:
            series = series.where(pd.to_datetime(history[receipt], utc=True).le(issue))
        source = f'{field}_spacecraft'
        if pd.notna(spacecraft) and source in history:
            series = series.where(history[source].eq(spacecraft))
        values[metric] = series
    frame = pd.DataFrame(values)
    frame['bs'] = (-frame.bz).clip(lower=0)
    recent = frame.loc[cutoff - 4 * MINUTE:cutoff]
    trailing = frame.loc[cutoff - 29 * MINUTE:cutoff]
    magnetic = ['bt', 'bz']
    if not recent[magnetic].count().ge(4).all() or not trailing[magnetic].count().ge(12).all():
        return None
    pressure = PRESSURE_FACTOR * recent.n * recent.v.pow(2)
    pressure_ratio = pressure.median() / trigger.baseline_pressure if pressure.notna().any() else np.nan
    means = {
        f'x_{metric}_mean_{width}m': float(frame.loc[cutoff - (width - 1) * MINUTE:cutoff, metric].mean())
        for metric in ('bt', 'bs', 'bz') for width in WINDOWS
    }
    return dict(
        means,
        x_bt_max_30m=float(trailing.bt.max()),
        x_bs_max_30m=float(trailing.bs.max()),
        x_bs10_fraction_30m=float(trailing.bs.dropna().ge(10).mean()),
        x_bt_baseline=float(trigger.baseline_bt),
        x_bt_jump=float(trigger.recent_bt - trigger.baseline_bt),
        x_speed_5m=float(recent.v.median()) if recent.v.notna().any() else np.nan,
        x_density_5m=float(recent.n.median()) if recent.n.notna().any() else np.nan,
        x_pressure_log_ratio=float(np.log(pressure_ratio)) if pressure_ratio > 0 else np.nan,
    )


class FastL1BtForecaster:
    registry_name = 'hmf_fast_l1_bt'

    def __init__(self, bundle):
        from lightgbm import Booster

        self.calibration = np.asarray(bundle.get('calibration'), dtype=float)
        if (bundle.get('kind') != 'tree' or tuple(bundle.get('features', ())) != FEATURES
                or self.calibration.shape != (2,) or not np.isfinite(self.calibration).all()
                or self.calibration[0] <= 0):
            raise ValueError('Invalid fast L1 Bt model')
        text = bundle.get('model_text')
        if not isinstance(text, str) or sha256(text.encode()).hexdigest() != bundle.get('model_sha256'):
            raise ValueError('Fast L1 Bt model hash mismatch')
        self.model = Booster(model_str=text)
        if self.model.num_feature() != len(FEATURES) + 1:
            raise ValueError('Fast L1 Bt model feature count mismatch')

    def forecast_frame(self, wind, *, trigger, issue_time):
        """Use measurements through the first detector crossing, on the native L1 clock."""
        from scipy.special import expit

        issue = _utc_issue(issue_time)
        cutoff = _utc_issue(trigger.trigger_time)
        if cutoff != cutoff.floor(MINUTE) or not MINUTE <= issue - cutoff <= MAX_TRIGGER_AGE:
            return None
        drivers = wind_features(wind, trigger, issue)
        if drivers is None:
            return None
        # Each full hour must finish within the six-hour horizon.
        targets = pd.date_range(issue.ceil(HOUR), issue + 5 * HOUR, freq=HOUR)
        frame = pd.DataFrame({**drivers, 'x_lead_hours': (targets - issue) / HOUR}, index=targets)
        values = frame[list(FEATURES)].to_numpy(dtype=np.float32)
        matrix = np.column_stack([np.repeat(values, len(THRESHOLDS), axis=0),
                                  np.tile(np.asarray(THRESHOLDS) / 5, len(frame))])
        raw = self.model.predict(matrix, raw_score=True, num_threads=1).reshape(len(frame), len(THRESHOLDS))
        scale, bias = self.calibration
        probabilities = expit(scale * raw + bias)
        if not np.isfinite(probabilities).all() or (np.diff(probabilities, axis=1) > 1e-10).any():
            raise ValueError('Invalid fast L1 Bt probabilities')
        return pd.DataFrame({'valid_time': targets, **{
            f'p_bt_ge_{threshold}': probabilities[:, i] for i, threshold in enumerate(THRESHOLDS)
        }})
