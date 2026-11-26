"""Short Kp/ap forecasts issued when a new front is detected at L1."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from forecast_core.inference.geomagnetic.ordinal import AP_LEVELS, KP_EXCEEDANCE, THRESHOLDS


MINUTE = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)
NATIVE = 3 * HOUR
WIND = ('v', 'n', 'bx', 'by', 'bz', 'bt')
DISTANCE_KM = 1_500_000
KP_DELAY = 4 * HOUR
DST_DELAY = 2 * HOUR
WIND_LOOKBACK = 4 * HOUR
MAX_TRIGGER_AGE = 31 * MINUTE
FEATURES = (
    'x_bt', 'x_bz', 'x_delta_bt', 'x_delta_speed', 'x_dst_latest', 'x_earth_lead_hours',
    'x_kp_latest', 'x_kp_recent_max', 'x_lead_hours', 'x_log_newell_30m',
    'x_log_newell_5m', 'x_log_pressure', 'x_pressure_ratio_log', 'x_south_fraction',
    'x_south_fraction_30m', 'x_speed',
)
PRESSURE_FACTOR = 1.6726e-6


@dataclass(frozen=True)
class FastL1Forecast:
    trigger_time: pd.Timestamp
    estimated_arrival: pd.Timestamp
    kp: pd.DataFrame
    ap: pd.DataFrame


def travel_time(speed):
    return (DISTANCE_KM / (60 * speed)).round()


def detect_l1_triggers(wind, *, start=None, end=None):
    """Scan a continuous UTC minute grid; missing vectors stay as gaps."""
    complete = wind[list(WIND)].notna().all(axis=1)
    pressure = PRESSURE_FACTOR * wind.n * wind.v ** 2
    driver = wind[['v', 'bt']].assign(pressure=pressure).where(complete)
    reference = driver.rolling(60, min_periods=40).median().shift(15)
    latest = driver.rolling(5, min_periods=4).median()
    eligible = complete & reference.notna().all(axis=1) & latest.notna().all(axis=1)
    if 'spacecraft' in wind:
        # Instrument changes are not jumps in the wind. Use a common source
        # across the two median windows whenever the provider supplies one.
        code = pd.Series(pd.factorize(wind.spacecraft)[0], index=wind.index)
        code = code.where(code.ge(0) & complete)
        recent_source = code.rolling(5, min_periods=4)
        reference_source = code.rolling(60, min_periods=40)
        eligible &= (code.eq(recent_source.min()) & code.eq(recent_source.max())
                     & code.eq(reference_source.min().shift(15))
                     & code.eq(reference_source.max().shift(15)))
    delta = latest - reference
    pressure_jump = latest.pressure.ge(1.8 * reference.pressure) & delta.pressure.ge(.8)
    field_jump = latest.bt.ge(1.4 * reference.bt) & delta.bt.ge(2)
    speed_jump = delta.v.ge(30)
    candidate = eligible & pressure_jump & (field_jump | speed_jump)
    edges = candidate.index[candidate & ~candidate.shift(fill_value=False)]
    selected = []
    for time in edges:
        if not selected or time - selected[-1] >= 90 * MINUTE:
            selected.append(time)
    trigger = latest.loc[selected].add_prefix('recent_').join(
        reference.loc[selected].add_prefix('baseline_'), validate='one_to_one',
    )
    trigger.index.name = 'trigger_time'
    if start is not None:
        trigger = trigger.loc[trigger.index >= start]
    if end is not None:
        trigger = trigger.loc[trigger.index < end]
    trigger = trigger.reset_index()
    trigger.insert(0, 'trigger_id', trigger.trigger_time.dt.strftime('l1-%Y%m%dT%H%M'))
    trigger['front_l1_time'] = trigger.trigger_time - 4 * MINUTE
    trigger['travel_minutes'] = travel_time(trigger.recent_v)
    trigger['estimated_arrival'] = trigger.front_l1_time + trigger.travel_minutes * MINUTE
    return trigger, eligible


def _mean_log_newell(window):
    angle = np.arctan2(window.by, window.bz)
    bt_transverse = np.hypot(window.by, window.bz)
    coupling = window.v.pow(4 / 3) * bt_transverse.pow(2 / 3) * np.sin(angle / 2).abs().pow(8 / 3)
    return float(np.log1p(coupling.mean()))


def wind_features(wind, trigger, update_minutes=0):
    cutoff = trigger.trigger_time + update_minutes * MINUTE
    last_30 = wind.loc[cutoff - 29 * MINUTE:cutoff].dropna(subset=list(WIND))
    if 'travel_ready' in last_30:
        last_30 = last_30.loc[last_30.travel_ready]
    if 'spacecraft' in last_30:
        last_30 = last_30.loc[last_30.spacecraft.eq(wind.at[trigger.trigger_time, 'spacecraft'])]
    last_5 = last_30.loc[max(cutoff - 4 * MINUTE, trigger.front_l1_time):]
    if len(last_5) < 4:
        return {}, len(last_5), len(last_30), 'insufficient_recent_wind'
    if len(last_30) < 12:
        return {}, len(last_5), len(last_30), 'insufficient_trailing_wind'
    pressure = float((PRESSURE_FACTOR * last_5.n * last_5.v ** 2).median())
    v = float(last_5.v.median())
    bt = float(last_5.bt.median())
    return {
        'x_speed': v,
        'x_bt': bt,
        'x_bz': float(last_5.bz.median()),
        'x_south_fraction': float(last_5.bz.le(-5).mean()),
        'x_log_pressure': float(np.log1p(pressure)),
        'x_log_newell_5m': _mean_log_newell(last_5),
        'x_log_newell_30m': _mean_log_newell(last_30),
        'x_south_fraction_30m': float(last_30.bz.le(-5).mean()),
        'x_delta_speed': v - trigger.baseline_v,
        'x_delta_bt': bt - trigger.baseline_bt,
        'x_pressure_ratio_log': float(np.log(pressure / trigger.baseline_pressure)),
    }, len(last_5), len(last_30), 'ready'


def index_features(kp, dst, issue):
    kp_end, dst_end = issue - KP_DELAY, issue - DST_DELAY
    kp_history = kp.loc[kp_end - 12 * HOUR:kp_end].dropna()
    dst_history = dst.loc[dst_end - 5 * HOUR:dst_end].dropna()
    return {
        'x_kp_latest': float(kp_history.iloc[-1]) if len(kp_history) else np.nan,
        'x_kp_recent_max': float(kp_history.max()),
        'x_dst_latest': float(dst_history.iloc[-1]) if len(dst_history) else np.nan,
    }, kp_history.index.max(), dst_history.index.max()


def _utc_index(index, interval):
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans
            or index.has_duplicates or not index.is_monotonic_increasing):
        raise ValueError('Expected sorted unique timezone-aware observations')
    clock = index.tz_convert('UTC')
    if not (clock == clock.floor(interval)).all():
        raise ValueError('Expected observations at native interval starts')
    return clock


def _index_series(series, interval, bounds):
    if series.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz='UTC'))
    values = series.astype(float)
    values.index = _utc_index(values.index, interval)
    return values.where(values.between(*bounds))


def _known_wind(wind, issue):
    clock = _utc_index(wind.index, MINUTE)
    columns = [*WIND, *(['spacecraft'] if 'spacecraft' in wind else [])]
    known = wind[columns].set_axis(clock)
    if 'received_at' in wind:
        received = pd.to_datetime(wind.received_at, utc=True).set_axis(clock)
        known = known.loc[received.le(issue)]
    speed = known.v
    if 'plasma_speed' in wind:
        speed = wind.plasma_speed.set_axis(clock)
        if 'plasma_received_at' in wind:
            received = pd.to_datetime(wind.plasma_received_at, utc=True).set_axis(clock)
            speed = speed.where(received.le(issue))
    grid = pd.date_range(issue.floor('min') - WIND_LOOKBACK,
                         issue.floor('min') - MINUTE, freq=MINUTE)
    known = known.reindex(grid)
    drivers = known[list(WIND)].astype(float)
    valid = (drivers.v.between(100, 2500) & drivers.n.gt(0) & drivers.n.le(150)
             & drivers.bt.ge(0) & drivers[['bx', 'by', 'bz', 'bt']].abs().lt(400).all(axis=1))
    known[list(WIND)] = drivers.where(valid)
    # The training windows retained complete vectors only when the short
    # plasma-only speed history could also provide a travel estimate.
    speed = speed.astype(float).reindex(grid)
    known['travel_ready'] = speed.where(speed.between(100, 2500)).rolling(5, min_periods=3).count().ge(3)
    return known


def _utc_issue(value):
    issue = pd.Timestamp(value)
    if pd.isna(issue) or issue.tz is None:
        raise ValueError('Expected a timezone-aware fast L1 issue')
    return issue.tz_convert('UTC')


def _detect_known(wind, issue):
    if wind.empty:
        return wind, None
    known = _known_wind(wind, issue)
    triggers, _ = detect_l1_triggers(known, start=issue - MAX_TRIGGER_AGE)
    return known, next(triggers.itertuples(index=False), None)


def find_trigger(wind, *, issue_time):
    """Find the recent candidate front before fetching geomagnetic history."""
    _, trigger = _detect_known(wind, _utc_issue(issue_time))
    return trigger


class FastL1GeomagneticForecaster:
    registry_name = 'geomagnetic_fast_l1'

    def __init__(self, bundle):
        from lightgbm import Booster

        self.calibration = np.asarray(bundle.get('calibration'), dtype=float)
        if (bundle.get('kind') != 'tree_ordinal' or tuple(bundle.get('features', ())) != FEATURES
                or self.calibration.shape != (2,) or not np.isfinite(self.calibration).all()
                or self.calibration[0] <= 0):
            raise ValueError('Invalid fast L1 geomagnetic model')
        text = bundle.get('model_text')
        if not isinstance(text, str) or sha256(text.encode()).hexdigest() != bundle.get('model_sha256'):
            raise ValueError('Fast L1 geomagnetic model hash mismatch')
        self.model = Booster(model_str=text)
        if self.model.num_feature() != len(FEATURES) + 1:
            raise ValueError('Fast L1 geomagnetic model feature count mismatch')

    def forecast_frames(self, wind, kp, dst, *, issue_time):
        """Return jointly distributed Kp/ap for complete bins after this issue.

        Inputs are the native observations available at issue time. Wind
        features are frozen at the detected trigger's sample minute.
        """
        from scipy.special import expit

        issue = _utc_issue(issue_time)
        known, trigger = _detect_known(wind, issue)
        if trigger is None:
            return None
        drivers, _, _, status = wind_features(known, trigger)
        if status != 'ready':
            return None
        geomagnetic, _, _ = index_features(_index_series(kp, NATIVE, (0, 9)),
                                            _index_series(dst, HOUR, (-2000, 1000)), issue)
        targets = pd.date_range(issue.ceil(NATIVE), issue + NATIVE, freq=NATIVE)
        frame = pd.DataFrame({
            **drivers, **geomagnetic, 'x_lead_hours': (targets - issue) / HOUR,
            'x_earth_lead_hours': (targets - trigger.estimated_arrival) / HOUR,
        }, index=targets)
        values = frame[list(FEATURES)].to_numpy(dtype=np.float32)
        matrix = np.column_stack([np.repeat(values, len(THRESHOLDS), axis=0),
                                  np.tile(THRESHOLDS / 3, len(frame))])
        raw = self.model.predict(matrix, raw_score=True, num_threads=1).reshape(len(frame), len(THRESHOLDS))
        scale, bias = self.calibration
        survival = expit(scale * raw + bias)
        if not np.isfinite(survival).all() or (np.diff(survival, axis=1) > 1e-10).any():
            raise ValueError('Invalid fast L1 ordinal distribution')
        labels = pd.DataFrame({'valid_time': targets})
        kp = labels.assign(**{f'p_kp_ge_{k}': survival[:, 3 * k - 1] for k in KP_EXCEEDANCE})
        ap = labels.assign(**{f'ap_q{int(100*q)}': AP_LEVELS[(survival > 1-q).sum(axis=1)]
                              for q in (.1, .5, .9)})
        return FastL1Forecast(trigger.trigger_time, trigger.estimated_arrival, kp, ap)
