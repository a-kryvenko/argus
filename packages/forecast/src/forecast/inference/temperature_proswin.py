"""Hourly temperature LightGBM heads with optional causal PROSWIN speed inputs."""
import logging
import numpy as np
import pandas as pd
from forecast.inference.density_proswin import history_features
from forecast.inference.proswin_blend import select_predictions

FORMAT = 'temperature_proswin'
VERSION = 'argus-plasma-temperature-proswin-v1'
HISTORY_NAMES = ('speed', 'age_hours', 'std_24h', 'count_24h', 'change_3h', 'change_6h', 'change_12h', 'change_24h')
FEATURES = {name: [f'{t}_{c}' for t in targets for c in HISTORY_NAMES] + extra for name, targets, extra in (
    ('t_history', ('t',), []), ('tnv_history', ('t', 'n', 'v'), []),
    ('tnv_proswin', ('t', 'n', 'v'), ['proswin']))}
logger = logging.getLogger(__name__)


def aware(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('Timezone-aware timestamp required')
    return result.tz_convert('UTC')


def hourly_histories(payload, *, issue_time, as_of):
    """Use closed Clio hourly means; preserve missing hours and receipt cutoff."""
    issue, cutoff = aware(issue_time), aware(as_of)
    if issue != issue.floor('h') or issue > cutoff:
        raise ValueError('Invalid temperature input cutoff')
    if not payload or payload.get('resolution_seconds') != 3600 or payload.get('gap_filling') != 'none':
        raise ValueError('Unfilled hourly Clio plasma history required')
    result = {}
    for target, unit in [('t', 'K'), ('n', 'cm⁻³'), ('v', 'km/s')]:
        series = payload.get('series', {}).get(target)
        rows = []
        if series is not None:
            if (series.get('unit') != unit or series.get('aggregation') != 'mean_min_max'
                    or series.get('location') != 'L1' or series.get('time_basis') != 'measurement'
                    or series.get('propagated') is not False):
                raise ValueError(f'Incompatible hourly plasma metadata: {target}')
            seen = set()
            for point in series['points']:
                start, end = aware(point['observed_at']), aware(point['interval_end'])
                if start in seen or start != start.floor('h') or end != start + pd.Timedelta(hours=1):
                    raise ValueError('Invalid or duplicate hourly plasma interval')
                seen.add(start)
                if end > issue or end < issue-pd.Timedelta(hours=48) or aware(point['received_at']) > cutoff:
                    continue
                if (point.get('quality') not in ('good', 'unverified') or point.get('recalculation_pending')
                        or point.get('expected_count') != 60 or not 0 < point.get('count', 0) <= 60):
                    continue
                value = point.get('value')
                if value is not None and np.isfinite(value):
                    # Training observations become available at the end of their hour.
                    rows.append({'issue_time': end, target: float(value)})
        result[target] = pd.DataFrame(rows, columns=['issue_time', target])
    return result


class TemperatureProswinForecaster:
    registry_name = 'plasma_temperature_quantile'
    uses_temperature_proswin = True

    def __init__(self, bundle):
        if bundle.get('format') != FORMAT or bundle.get('version') != 1 or bundle.get('features') != FEATURES:
            raise ValueError('Unsupported temperature artifact')
        import lightgbm as lgb
        self.models_bundle = bundle
        self._prediction_error = lgb.basic.LightGBMError
        self.heads, self.offsets = {}, {}
        for name, features in FEATURES.items():
            self.offsets[name] = np.asarray(bundle['interval_offsets'][name], float)
            if (len(bundle['heads'][name]) != 96 or self.offsets[name].shape != (96, 2)
                    or not np.isfinite(self.offsets[name]).all()
                    or (self.offsets[name][:, 0] > self.offsets[name][:, 1]).any()):
                raise ValueError('Invalid temperature heads or intervals')
            self.heads[name] = [lgb.Booster(model_str=model) for model in bundle['heads'][name]]
            if any(head.feature_name() != features for head in self.heads[name]):
                raise ValueError('Temperature feature order mismatch')
        self.last_status = {}

    def forecast_snapshot(self, inputs, *, issue_time):
        histories = hourly_histories(inputs.solar_wind_hourly, issue_time=issue_time, as_of=inputs.as_of)
        return self.frame(issue_time, histories, inputs.proswin_predictions,
                          ready_at=inputs.proswin_ready_at or inputs.as_of, source_cutoff=inputs.as_of)

    def frame(self, issue_time, histories, predictions=(), *, ready_at=None, source_cutoff=None):
        issue = aware(issue_time)
        if issue != issue.floor('h'):
            raise ValueError('Issue must be an hourly boundary')
        row = {}
        for target in ('t', 'n', 'v'):
            row.update(history_features(issue, histories[target], target))
        if not np.isfinite(row['t_speed']):
            raise ValueError('No temperature observation available within two hours')
        frame = pd.DataFrame({'issue_time': issue, 'lead_hours': np.arange(1, 97)})
        frame['valid_time'] = issue + pd.to_timedelta(frame.lead_hours, unit='h')
        speed = select_predictions(issue, frame.valid_time, predictions, ready_at=ready_at, source_cutoff=source_cutoff)
        have_plasma = np.isfinite([row['n_speed'], row['v_speed']]).all()
        used = {name: 0 for name in FEATURES}
        for i in range(96):
            candidates = (['tnv_proswin'] if have_plasma and np.isfinite(speed[i]) else [])
            candidates += (['tnv_history'] if have_plasma else []) + ['t_history']
            x = pd.DataFrame([{**row, 'proswin': speed[i]}]).astype(np.float32)
            for name in candidates:
                try:
                    point = float(self.heads[name][i].predict(x[FEATURES[name]], num_threads=1)[0])
                    if not np.isfinite(point):
                        raise ValueError('Nonfinite temperature prediction')
                    point = max(point, 0.)
                    low, high = self.offsets[name][i]
                    frame.loc[i, ['t_q10', 't_q50', 't_q90']] = [max(min(point+low, point), 0), point, max(point+high, point)]
                    used[name] += 1
                    break
                except (ValueError, RuntimeError, self._prediction_error):
                    if name == 't_history':
                        raise
                    logger.exception('Temperature %s failed at +%dh; trying fallback', name, i+1)
        self.last_status = {'candidate_leads': used['tnv_proswin'],
                            'fallback_leads': used['tnv_history'] + used['t_history'],
                            'history_fallback_leads': used['tnv_history'], 'temperature_only_leads': used['t_history']}
        return frame
