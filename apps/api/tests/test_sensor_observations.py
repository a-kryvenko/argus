from datetime import UTC, datetime, timedelta

import pandas as pd

from app.db.models import Measurement, NormalizedObservation
from app.services.sensor_observations import (
    OBSERVATION_METRICS,
    normalize_measurements,
)


def _measurement_rows(timestamp: datetime, offset: float = 0) -> list[dict]:
    return [
        {
            "observed_at": timestamp,
            "metric": metric,
            "value": index + offset,
        }
        for index, metric in enumerate(OBSERVATION_METRICS)
    ]


def test_normalize_measurements_builds_hourly_wide_rows() -> None:
    start = datetime(2026, 8, 30, tzinfo=UTC)
    measurements = pd.DataFrame([
        *_measurement_rows(start),
        *_measurement_rows(start + timedelta(hours=2), offset=2),
    ])

    result = normalize_measurements(measurements)

    assert list(result.columns) == ["observed_at", *OBSERVATION_METRICS]
    assert list(result["observed_at"]) == list(pd.date_range(start, periods=3, freq="1h"))
    middle = result.iloc[1]
    assert middle["bx"] == 1
    assert middle["f10_7"] == 10


def test_normalize_measurements_requires_every_metric() -> None:
    timestamp = datetime(2026, 8, 30, tzinfo=UTC)
    measurements = pd.DataFrame(_measurement_rows(timestamp))
    measurements = measurements[measurements["metric"] != "dst"]

    result = normalize_measurements(measurements)

    assert result.empty


def test_database_models_match_narrow_and_wide_storage_contracts() -> None:
    assert set(Measurement.__table__.columns.keys()) == {
        "id",
        "metric",
        "value",
        "observed_at",
    }
    assert set(NormalizedObservation.__table__.columns.keys()) == {
        "observed_at",
        *OBSERVATION_METRICS,
    }
