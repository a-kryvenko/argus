"""Causal driver preparation used exclusively by the JB2008 forecast.

DTC equations follow DTCAP/DSTDTC in Bowman's DTCMAKEDR_AUTO.f (2023
release): https://sol.spacenvironment.net/JB2008/downloadables/jb2008.zip.
The reference event detector needs future Dst. Our online event detection and
running storm minimum are approximations, identified as causal_dst_ap_v1.
The solar background is a trailing daily mean, NOT a centered observed mean.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

SOLAR_LAGS_DAYS = {"f10_7": 1, "s10": 1, "m10": 2, "y10": 5}
SOURCE_METRICS = (*SOLAR_LAGS_DAYS, "dst", "ap")
BACKGROUND_METHOD = "trailing_81_daily_values"
DTC_METHOD = "causal_dst_ap_v1"
AP_LAG = pd.Timedelta(days=0.279)  # Reference DTCAP: 6.696 hours.


class DriverDataUnavailable(ValueError):
    """Insufficient or invalid observations for a JB2008 driver snapshot."""


@dataclass(frozen=True)
class DriverSnapshot:
    values: dict[str, float]
    observed_at: pd.Timestamp
    history_start: pd.Timestamp
    dtc_observed_at: pd.Timestamp
    background_interpolated_days: dict[str, list[str]]


def fill_background_gaps(daily: pd.Series) -> tuple[pd.Series, list[str]]:
    """Fill at most two missing interior days in total, exclusively for a mean.

    The daily index is a contiguous UTC calendar grid. No extrapolation,
    nonpositive/infinite value replacement, or partial filling of longer gaps.
    """
    missing = daily.isna()
    if (not 0 < int(missing.sum()) <= 2 or missing.iloc[0] or missing.iloc[-1]
            or not np.isfinite(daily[~missing]).all() or not daily[~missing].gt(0).all()):
        return daily, []
    filled = daily.interpolate(method="time", limit_area="inside")
    return filled, [day.date().isoformat() for day in daily.index[missing]]


def quiet_dtc(ap: float) -> float:
    """Reference Jacchia-70 quiet-time correction in K, with ap capped at 50."""
    if not np.isfinite(ap) or not 0 <= ap <= 400:
        raise DriverDataUnavailable("Invalid ap observation")
    activity = min(ap, 50.)
    return float(activity - 100. * np.expm1(-0.08 * activity))


def _main_slope(minimum: float) -> float:
    return -1.4 if minimum < -450 else -1.505e-5 * minimum**2 - 1.0604e-2 * minimum - 3.2


def _main_step(temperature: float, previous: float, current: float, minimum: float) -> float:
    previous, current = min(previous, 0.), min(current, 0.)
    slope = _main_slope(minimum)
    change = current - previous
    if change >= 0:
        return temperature - 0.3 * slope * change
    return 0.846 * temperature + slope * (current - 0.870 * previous)


def causal_dtc(dst: pd.Series, lagged_ap: pd.Series) -> pd.Series:
    """Hourly DTC estimate; each output depends only on this and earlier rows.

    Initialize after six quiet Dst hours. A drop >=50 nT reaching -75 triggers
    a storm; replay its observed descent from the preceding maximum. Use the
    running minimum in the published main-phase slope and 0/1/2-hour lag.
    Three rising hours confirm recovery; three shallow two-hour slopes
    (<100 nT/day) switch to late recovery. Six hours above -75 end the event.
    These online event rules are not the reference's retrospective segmentation.
    """
    if (len(dst) < 72 or not dst.index.equals(lagged_ap.index)
            or not isinstance(dst.index, pd.DatetimeIndex)
            or not dst.index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
            or not np.isfinite(dst).all() or not np.isfinite(lagged_ap).all()
            or not dst.between(-2000, 100).all()):
        raise DriverDataUnavailable("DTC requires at least 72 consecutive valid hourly Dst/ap values")
    quiet = np.array([quiet_dtc(value) for value in lagged_ap])
    values = dst.to_numpy(dtype=float)
    output = quiet.copy()
    integrated = quiet.copy()
    initialized = False
    quiet_hours = 0
    active = False
    peak = 0
    minimum = 0.
    minimum_at = 0
    rising = shallow = recovered = 0
    phase = "main"
    temperature = quiet[0]
    for i, current in enumerate(values):
        quiet_hours = quiet_hours + 1 if current >= -40 else 0
        if not initialized:
            if quiet_hours >= 6:
                initialized = True
                peak = i
            else:
                output[i] = np.nan
                continue
        if not active:
            if current >= values[peak] or quiet_hours >= 6:
                peak = i
            if current > -75 or values[peak] - current < 50:
                continue
            active = True
            minimum, minimum_at = current, i
            phase = "main"
            rising = shallow = recovered = 0
            temperature = quiet[peak]
            # No future values: onset is recognized only once the threshold is reached.
            for j in range(peak + 1, i + 1):
                temperature = _main_step(temperature, values[j-1], values[j], minimum)
                integrated[j] = max(0., temperature)
        else:
            change = current - values[i-1]
            if current < minimum:
                minimum, minimum_at = current, i
                phase = "main"
                rising = shallow = 0
            rising = rising + 1 if change > 0 else 0
            if phase == "main" and rising >= 3:
                phase = "recovery"
            if phase == "recovery" and i >= minimum_at + 2:
                slope = (current - values[i-2]) * 12
                shallow = shallow + 1 if 0 <= slope < 100 else 0
                if shallow >= 3:
                    phase = "late_recovery"
            if phase == "main":
                temperature = _main_step(temperature, values[i-1], current, minimum)
            elif phase == "recovery":
                temperature += 0.13 * current
            else:
                temperature += (-2.5 if change >= 0 else _main_slope(minimum)) * change
            temperature = max(0., temperature)
            integrated[i] = temperature
        delay = 0 if minimum <= -350 else 1 if minimum <= -250 else 2
        output[i] = integrated[max(peak, i-delay)]
        recovered = recovered + 1 if current > -75 else 0
        duration_hours = max(1, int(0.0075 * (values[peak] - minimum) * 24))
        if (recovered >= 6 or temperature <= 0
                or (phase != "main" and i - minimum_at >= duration_hours)):
            active = False
            peak = i
            output[i] = quiet[i]
    if not np.isfinite(output[-1]):
        raise DriverDataUnavailable("DTC history has no six-hour quiet initialization interval")
    return pd.Series(output, index=dst.index, name="dtc")


def prepare_snapshot(measurements: pd.DataFrame, issue_time) -> DriverSnapshot:
    issue = pd.Timestamp(issue_time)
    if issue.tzinfo is None:
        raise ValueError("issue_time must include a timezone")
    records = measurements.copy()
    records["observed_at"] = pd.to_datetime(records["observed_at"], utc=True)
    records = records.loc[records.observed_at.le(issue)].sort_values("observed_at")
    records["value"] = pd.to_numeric(records["value"], errors="coerce")
    records = records.drop_duplicates(["metric", "observed_at"], keep="last")
    snapshot = {}
    timestamps = []
    starts = []
    interpolated_days = {}
    for name, lag in SOLAR_LAGS_DAYS.items():
        cutoff = issue - pd.Timedelta(days=lag)
        solar = records.loc[records.metric.eq(name) & records.observed_at.le(cutoff)]
        if solar.empty or cutoff - solar.observed_at.iloc[-1] > pd.Timedelta(hours=48):
            raise DriverDataUnavailable(f"Missing recent {name} observations")
        # Last snapshot per UTC day: hourly provisional solar estimates do not
        # give well-sampled days more weight than sparsely sampled days.
        daily = solar.set_index("observed_at").value.resample("D").agg(lambda day: day.iloc[-1] if len(day) else np.nan)
        days = pd.date_range(daily.index[-1] - pd.Timedelta(days=80), daily.index[-1], freq="D")
        daily = daily.reindex(days)
        # Explicit invalid observations are rejected; only absent calendar days
        # may be interpolated. Never modify the raw solar observations.
        present_days = solar.observed_at.dt.floor("D")
        if not (daily.isna() & daily.index.isin(present_days)).any():
            daily, filled_days = fill_background_gaps(daily)
            if filled_days:
                interpolated_days[name] = filled_days
        if not np.isfinite(daily).all() or not daily.gt(0).all():
            valid_days = np.isfinite(daily) & daily.gt(0)
            missing_days = days[~valid_days.to_numpy()]
            raise DriverDataUnavailable(
                f"{name} requires 81 consecutive valid daily observations; "
                f"available={int(valid_days.sum())}/81, "
                f"missing={missing_days[0].date()}..{missing_days[-1].date()}"
            )
        snapshot[name] = float(daily.iloc[-1])
        snapshot[f"{name}_81mean"] = float(daily.mean())
        timestamps.append(solar.observed_at.iloc[-1])
        starts.append(days[0])

    dst = records.loc[records.metric.eq("dst")].set_index("observed_at").value
    ap = records.loc[records.metric.eq("ap")].set_index("observed_at").value
    if dst.empty or ap.empty or issue - dst.index[-1] > pd.Timedelta(hours=3):
        raise DriverDataUnavailable("Missing recent Dst/ap observations")
    end = dst.index[-1].floor("h")
    hours = pd.date_range(end - pd.Timedelta(hours=167), end, freq="h")
    # Only exact completed hourly samples for Dst. Missing hours are not
    # interpolated through a possible storm.
    hourly_dst = dst.loc[dst.index <= end].reindex(hours)
    # ap is a 3-hour index; as-of selection implements the reference 6.696h lag.
    ap_times = hours - AP_LAG
    lagged_ap = ap.reindex(ap_times, method="ffill", tolerance=pd.Timedelta(hours=3))
    lagged_ap.index = hours
    valid = np.isfinite(hourly_dst) & np.isfinite(lagged_ap)
    gaps = np.flatnonzero(~valid.to_numpy())
    start = int(gaps[-1] + 1) if len(gaps) else 0
    hourly_dst, lagged_ap = hourly_dst.iloc[start:], lagged_ap.iloc[start:]
    series = causal_dtc(hourly_dst, lagged_ap)
    snapshot["dtc"] = float(series.iloc[-1])
    timestamps.append(end)
    return DriverSnapshot(snapshot, min(timestamps), min(starts), end, interpolated_days)
