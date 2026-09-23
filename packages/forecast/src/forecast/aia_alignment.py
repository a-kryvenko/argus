"""Causal fixed96h arrival window; latest image for horizons96–120."""
import numpy as np
import pandas as pd
from forecast.aia_schema import feature_columns
HOUR=3600*10**9

def align(frame, solar, plan):
    """No target column is consulted. All times compared in integer UTC nanoseconds."""
    observed = pd.DatetimeIndex(solar.observed_at).as_unit('ns').asi8
    available = pd.DatetimeIndex(solar.available_at).as_unit('ns').asi8
    if not ((np.diff(observed) > 0).all() and (np.diff(available) > 0).all() and (available >= observed).all()):
        raise ValueError('Noncausal or unordered source observations')
    columns = feature_columns(solar)
    solar_columns = [c for c in columns if c not in ['dlinear_v', 'lead_hours', 'calendar_sin', 'calendar_cos', 'aia_age_hours']]
    source = solar[solar_columns].to_numpy(dtype=float)
    if np.isinf(source).any():
        raise ValueError('Infinite source feature')
    output = frame[['issue_time', 'valid_time', 'lead_hours', 'dlinear_v', 'target_v', 'calendar_sin', 'calendar_cos']].copy()
    matrix = np.full((len(frame), len(solar_columns)), np.nan)
    ages = np.full(len(frame), np.nan)
    counts = np.zeros(len(frame), dtype=int)
    delays = np.zeros(len(frame))
    centers = np.zeros(len(frame))
    future_fraction = np.zeros(len(frame))
    for begin in range(0, len(frame), 4096):
        part = frame.iloc[begin:begin + 4096]
        n = len(part)
        issue = pd.DatetimeIndex(part.issue_time).as_unit('ns').asi8
        valid = pd.DatetimeIndex(part.valid_time).as_unit('ns').asi8
        lead = part.lead_hours.to_numpy()
        if not np.array_equal(valid, issue + lead * HOUR):
            raise ValueError('Incorrect valid_time')
        last = np.searchsorted(available, issue, side='right') - 1
        safe = np.maximum(last, 0)
        last_age = (issue - observed[safe]) / HOUR
        usable = (last >= 0) & (last_age <= plan['max_latest_age_hours'])
        tau = np.full(n, float(plan['fixed_delay_hours']))
        desired = valid - np.rint(tau * HOUR).astype(np.int64)
        center = np.minimum(desired, observed[safe])
        latest = lead >= plan['latest_from_horizon']
        center = np.where(latest, observed[safe], center)
        sums = np.zeros((n, len(solar_columns)))
        denominator = np.zeros_like(sums)
        age_sum = np.zeros(n)
        weight_sum = np.zeros(n)
        count = np.zeros(n, dtype=int)
        after_issue_weight = np.zeros(n)
        radius = plan['window_radius_hours'] * HOUR
        lo = np.searchsorted(observed, center - radius, side='left')
        hi = np.minimum(np.searchsorted(observed, center + radius, side='right'), last + 1)
        candidates = [(safe, usable & latest)]
        for offset in range(max(0, int(np.max(hi - lo)))):
            index = lo + offset
            candidates.append((np.clip(index, 0, len(observed) - 1), usable & ~latest & (index < hi)))
        for index, eligible in candidates:
            eligible &= available[index] <= issue
            weight = np.where(eligible, np.exp(-0.5 * ((observed[index] - center) / HOUR / plan['window_sigma_hours']) ** 2), 0.0)
            values = source[index]
            finite = np.isfinite(values)
            sums += np.where(finite, values, 0.0) * weight[:, None]
            denominator += finite * weight[:, None]
            age_sum += (issue - observed[index]) / HOUR * weight
            weight_sum += weight
            count += eligible
            after_issue_weight += weight * (observed[index] > issue)
        with np.errstate(invalid='ignore', divide='ignore'):
            matrix[begin:begin + n] = np.where(denominator > 0, sums / denominator, np.nan)
            ages[begin:begin + n] = np.where(weight_sum > 0, age_sum / weight_sum, np.nan)
        counts[begin:begin + n] = count
        delays[begin:begin + n] = tau
        centers[begin:begin + n] = (issue - center) / HOUR
        future_fraction[begin:begin + n] = after_issue_weight
    if future_fraction.any():
        raise ValueError('Future observation used')
    for j, column in enumerate(solar_columns):
        output[column] = matrix[:, j]
    output['aia_age_hours'] = ages
    output['aia_frame_index'] = 0
    output['aia_available'] = counts > 0
    output['selected_frames'] = counts
    output['propagation_hours'] = delays
    output['center_age_hours'] = centers
    return output
