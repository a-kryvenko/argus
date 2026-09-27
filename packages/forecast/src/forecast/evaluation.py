"""Shared scoring of frozen predictions against observed, never filled targets."""
import numpy as np
import pandas as pd

from common.schemas.forecast_release import PREDICTION_COLUMNS


TARGETS = {
    'plasma_speed_quantile': 'v', 'plasma_speed_threshold': 'v',
    'plasma_density_quantile': 'n', 'kp_threshold': 'kp', 'ap_quantile': 'ap',
    'dst_quantile': 'dst', 'hmf_total_threshold': 'bt', 'hmf_southward_threshold': 'bs',
    'f10_7_quantile': 'f107', 's10_quantile': 's10',
    'm10_quantile': 'm10', 'y10_quantile': 'y10',
}


def utc_column(values):
    """Reject ambiguous naive times instead of silently assigning UTC."""
    if any(pd.Timestamp(value).tzinfo is None for value in values):
        raise ValueError('All timestamps must include a timezone')
    return pd.to_datetime(values, utc=True, errors='raise')


def validate_predictions(frame, artifact):
    columns = ['issue_time', 'valid_time', 'lead_hours', *PREDICTION_COLUMNS[artifact]]
    result = frame[columns].copy()
    if result.empty:
        raise ValueError('No prediction rows')
    for name in ('issue_time', 'valid_time'):
        result[name] = utc_column(result[name])
        if result[name].isna().any():
            raise ValueError('Missing forecast timestamp')
    leads = pd.to_numeric(result.lead_hours, errors='raise')
    if not (np.isfinite(leads) & leads.ge(0) & leads.eq(np.floor(leads))).all():
        raise ValueError('Invalid lead_hours')
    result['lead_hours'] = leads.astype(int)
    if not (result.valid_time == result.issue_time + pd.to_timedelta(leads, unit='h')).all():
        raise ValueError('valid_time must equal issue_time + lead_hours')
    if result.duplicated(['issue_time', 'lead_hours']).any():
        raise ValueError('Duplicate forecast keys')
    predictions = result[list(PREDICTION_COLUMNS[artifact])].to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite predictions')
    if PREDICTION_COLUMNS[artifact][0].startswith('p_'):
        if ((predictions < 0) | (predictions > 1)).any():
            raise ValueError('Probabilities must be between zero and one')
    elif (np.diff(predictions, axis=1) < 0).any():
        raise ValueError('Quantiles must be ordered')
    return result


def match_observations(predictions, observations, *, as_of):
    """Match exact UTC hour starts; a target is due only after its hour closes."""
    columns = ['valid_time', 'value'] + (['sample_count'] if 'sample_count' in observations else [])
    truth = observations[columns].copy()
    truth['valid_time'] = utc_column(truth.valid_time)
    if truth.valid_time.isna().any() or truth.valid_time.duplicated().any():
        raise ValueError('Observation timestamps must be present and unique')
    truth['value'] = pd.to_numeric(truth.value, errors='raise')
    result = predictions.merge(truth, on='valid_time', how='left', validate='many_to_one')
    due = result.valid_time + pd.Timedelta(hours=1) <= pd.Timestamp(as_of)
    finite = np.isfinite(result.value.to_numpy(dtype=float))
    result['state'] = np.where(~due, 'pending', np.where(finite, 'verified', 'missing'))
    result.loc[~due, 'value'] = np.nan
    return result


def _binary(y, probability):
    from sklearn.metrics import average_precision_score, roc_auc_score
    predicted = probability >= .5
    tp, fp = int((y & predicted).sum()), int((~y & predicted).sum())
    fn, tn = int((y & ~predicted).sum()), int((~y & ~predicted).sum())
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    bins = np.minimum((probability * 10).astype(int), 9)
    reliability = []
    for bucket in range(10):
        mask = bins == bucket
        if mask.any():
            reliability.append(f'{probability[mask].mean():.6f}_{y[mask].mean():.6f}')
    return {
        'brier': float(np.square(probability - y).mean()),
        'roc_auc': float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None,
        'avg_precision': float(average_precision_score(y, probability)) if y.any() else None,
        'threat_score': tp / (tp + fp + fn) if tp + fp + fn else None,
        'heidke': 2 * (tp * tn - fp * fn) / denominator if denominator else None,
        'reliability': ';'.join(reliability),
    }


def score(frame, artifact):
    """Return website-compatible tables, plus explicit coverage denominators."""
    target = TARGETS[artifact]
    columns = PREDICTION_COLUMNS[artifact]
    tables = {}
    counts = {state: int(frame.state.eq(state).sum()) for state in ('verified', 'pending', 'missing')}
    counts['total'] = len(frame)
    for lead, group in frame.groupby('lead_hours', sort=True):
        part = group[group.state.eq('verified')]
        if part.empty:
            continue
        y = part.value.to_numpy(dtype=float)
        common = {'lead_hours': int(lead), 'n': len(part), 'scheduled': len(group)}
        if columns[0].startswith('p_'):
            for column in columns:
                threshold = float(column.rsplit('_', 1)[1])
                row = {**common, **_binary(y >= threshold, part[column].to_numpy(dtype=float))}
                tables.setdefault(f'threshold_{threshold:g}.csv', []).append(row)
        else:
            q10, q50, q90 = (part[f'{target}_q{q}'].to_numpy(dtype=float) for q in (10, 50, 90))
            error = q50 - y
            row = {**common, 'mae': float(abs(error).mean()),
                   'rmse': float(np.sqrt(np.square(error).mean())), 'bias': float(error.mean()),
                   'coverage_80': float(((y >= q10) & (y <= q90)).mean()),
                   'lower_tail': float((y < q10).mean()), 'upper_tail': float((y > q90).mean()),
                   'interval_width_80': float((q90 - q10).mean())}
            for q, prediction in ((.1, q10), (.5, q50), (.9, q90)):
                residual = y - prediction
                row[f'q{int(q * 100)}_pinball'] = float(np.maximum(q * residual, (q - 1) * residual).mean())
            tables.setdefault('regression.csv', []).append(row)
    return {'counts': counts, 'tables': tables}
