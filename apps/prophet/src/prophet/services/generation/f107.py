"""Day-ahead F10.7 forecasts using received Penticton measurements."""

from datetime import UTC, date, timedelta
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, FiniteFloat, model_validator


FEATURE = Literal['trend3', 'trend', 'week', 'rotation', 'level']
HORIZON_HOURS = 120


class DayCoefficients(BaseModel):
    intercept: FiniteFloat
    weights: dict[FEATURE, FiniteFloat]
    residual_offsets: tuple[FiniteFloat, FiniteFloat]
    persistence_offsets: tuple[FiniteFloat, FiniteFloat]

    @model_validator(mode='after')
    def validate_intervals(self):
        if any(not low <= 0 <= high for low, high in (self.residual_offsets, self.persistence_offsets)):
            raise ValueError('F10.7 interval offsets must bracket zero')
        return self


class DailyCoefficients(BaseModel):
    schema_version: Literal[1]
    max_age_days: int = Field(ge=0, le=3)
    days: dict[int, DayCoefficients]

    @model_validator(mode='after')
    def validate_horizon(self):
        if set(self.days) != set(range(1, HORIZON_HOURS // 24 + self.max_age_days + 1)):
            raise ValueError('Missing daily F10.7 forecast horizons')
        return self


def clip(value, low, high):
    return min(high, max(low, value))


class DailyF107Forecaster:
    registry_name = 'f10_7_quantile'

    def __init__(self, bundle):
        self.coefficients = DailyCoefficients.model_validate(bundle)

    def quantiles(self, observations: dict[date, float], target_date: date):
        latest_date = max(observations)
        latest = observations[latest_date]
        horizon = (target_date - latest_date).days
        if horizon == 0:
            return latest, latest, latest
        params = self.coefficients.days[horizon]

        def lag(days):
            return observations.get(latest_date - timedelta(days=days))

        lag_one, lag_three = lag(1), lag(3)
        week = [lag(day) for day in range(1, 8)]
        rotation_now, rotation_target = lag(27), lag(27 - horizon)
        features = {
            'trend3': clip(latest - lag_three, -40, 40) if lag_three is not None else None,
            'trend': clip(latest - lag_one, -35, 35) if lag_one is not None else None,
            'week': clip(sum(week) / 7 - latest, -40, 40) if None not in week else None,
            'rotation': clip(rotation_target - rotation_now, -40, 40)
                        if rotation_target is not None and rotation_now is not None else None,
            'level': clip(latest - 120, -100, 150),
        }
        if any(features[name] is None for name in params.weights):
            median = latest
            low, high = params.persistence_offsets
        else:
            correction = params.intercept + sum(weight * features[name] for name, weight in params.weights.items())
            median = latest + clip(correction, -40, 40)
            low, high = params.residual_offsets
        median = clip(median, 40, 500)
        return clip(median + low, 40, 500), median, clip(median + high, 40, 500)

    def forecast(self, inputs, *, issue_time):
        points = sorted(inputs.f107_measurements, key=lambda row: (row.observed_at, row.received_at))
        received = {point.observed_at.astimezone(UTC).date(): point for point in points
                    if point.observed_at <= point.received_at <= inputs.as_of}
        latest = max(received, default=None)
        if latest is None or (issue_time.date() - latest).days > self.coefficients.max_age_days:
            return None, {}

        observations = {day: point.value for day, point in received.items()}
        valid_times = [issue_time + timedelta(hours=lead) for lead in range(1, HORIZON_HOURS + 1)]
        daily = {day: self.quantiles(observations, day) for day in {stamp.date() for stamp in valid_times}}
        frame = pd.DataFrame([daily[stamp.date()] for stamp in valid_times],
                             columns=['f107_q10', 'f107_q50', 'f107_q90'])
        frame.insert(0, 'valid_time', valid_times)
        frame.insert(0, 'lead_hours', range(1, HORIZON_HOURS + 1))
        frame.insert(0, 'issue_time', issue_time)
        observation = received[latest]
        return frame, {'input_cutoff': inputs.as_of.isoformat(),
                       'last_observation_at': observation.observed_at.isoformat(),
                       'last_observation_received_at': observation.received_at.isoformat(),
                       'last_observation_source': observation.source_product}


def calculate_f107(inputs, *, issue_time, workdir, registry, legacy_service):
    from forecast.calculation import ForecastResult, calculate_snapshot
    from prophet.services.generation.models import load_model

    name = DailyF107Forecaster.registry_name
    entry = registry.get(name, {})
    if 'fallback' not in entry:
        service, metadata = load_model(legacy_service, workdir=workdir, registry=registry)
        return calculate_snapshot(service, inputs, issue_time=issue_time, model_info=metadata)

    service, metadata = load_model(DailyF107Forecaster, workdir=workdir, registry=registry)
    frame, observed = service.forecast(inputs, issue_time=issue_time)
    if frame is not None:
        return ForecastResult(name, frame, {**metadata, **observed})

    fallback, legacy = load_model(legacy_service, workdir=workdir, registry={name: entry['fallback']})
    legacy.update(fallback_for=metadata['model'], input_cutoff=inputs.as_of.isoformat())
    return calculate_snapshot(fallback, inputs, issue_time=issue_time, model_info=legacy)
