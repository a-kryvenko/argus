from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from forecast.observation_inputs import SourceMeasurement
from forecast.inputs import ForecastInputs
from common.schemas.observation import Observation
from prophet.services import solar_indices

NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


@pytest.mark.anyio
async def test_reads_only_solar_measurements_and_bounds_history(monkeypatch):
    items = [dict(metric=metric, value=100., observed_at=(NOW-timedelta(days=day)).isoformat())
             for metric in ('f10_7', 's10', 'm10', 'y10', 'dst') for day in (15, 3, 0)]
    row = {'as_of': NOW.isoformat(), 'measurements': items}
    result = Mock()
    result.mappings.return_value.first.return_value = row
    session = Mock(execute=AsyncMock(return_value=result))
    @asynccontextmanager
    async def connection():
        yield session
    monkeypatch.setattr(solar_indices, 'get_db_session', connection)
    data = await solar_indices.read_solar_indices()
    assert data.as_of == NOW
    assert len(data.items) == 8
    assert {point.metric for point in data.items} == {'f10_7', 's10', 'm10', 'y10'}
    result.mappings.return_value.first.return_value = None
    assert await solar_indices.read_solar_indices() is None


@pytest.mark.parametrize('metric', ['s10', 'm10', 'y10'])
def test_daily_model_matches_training_features_and_calibration_without_future_inputs(metric):
    from forecast_core.inference.radiation.daily import (
        FEATURES, DailyS10QuantileFS, DailyM10QuantileFS, DailyY10QuantileFS)
    models = [Mock(predict=Mock(return_value=np.full(5, value))) for value in (30., 10., 20.)]
    bundle = dict(target=metric, features=FEATURES, leads=(24, 48, 72, 96, 120),
                  models=models, offsets={lead: [1., 2., 3.] for lead in (24, 48, 72, 96, 120)})
    service = {'s10': DailyS10QuantileFS, 'm10': DailyM10QuantileFS, 'y10': DailyY10QuantileFS}[metric](bundle)
    last = NOW.replace(hour=0)-timedelta(days=2)
    rows = [SourceMeasurement(metric=metric, value=float(100-day), observed_at=last-timedelta(days=day))
            for day in range(28)]
    rows += [SourceMeasurement(metric=metric, value=999., observed_at=NOW),
             SourceMeasurement(metric=metric, value=888., observed_at=last+timedelta(hours=1),
                               available_at=NOW+timedelta(hours=1))]
    inputs = ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[]), measurements=rows)
    result = service.forecast_snapshot(inputs, issue_time=NOW)
    features = models[0].predict.call_args.args[0]
    assert features.iloc[0].tolist() == [100., 99., 98., 93., 73., 97., 87., 24.]
    assert result[f'{metric}_q10'].tolist() == [11.]*5
    assert result[f'{metric}_q50'].tolist() == [22.]*5
    assert result[f'{metric}_q90'].tolist() == [33.]*5
    assert result.valid_time.tolist() == [NOW+timedelta(hours=lead) for lead in bundle['leads']]
    inputs.measurements = []
    with pytest.raises(ValueError, match='Missing'):
        service.forecast_snapshot(inputs, issue_time=NOW)
