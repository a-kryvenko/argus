"""Receipt-gated F10.7 inference and fallback use the existing artifact contract."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import yaml

from common.schemas.forecast_release import ForecastArtifact
from common.schemas.observation import Observation
from forecast.inputs import F107Measurement, ForecastInputs
from prophet.services.generation import calculation, models
from prophet.services.generation.f107 import DailyF107Forecaster
from prophet.services.generation.products import PRODUCTS


ISSUE = datetime(2026, 11, 26, 22, tzinfo=UTC)
ARTIFACT = 'f10_7_quantile'
ROOT = Path(__file__).resolve().parents[3]


def point(days, value, *, receipt=None, source='swpc.f107'):
    observed = ISSUE - timedelta(days=days, hours=2)
    return F107Measurement(observed_at=observed, received_at=receipt or observed+timedelta(hours=1),
                           value=value, source_product=source)


@pytest.fixture
def runtime(monkeypatch):
    registry = yaml.safe_load((ROOT/'configs/models_registry.yaml').read_text())['models']
    product = PRODUCTS['solar-radiation']
    legacy_service = product.models[0].service_class()
    monkeypatch.setitem(PRODUCTS, 'solar-radiation', replace(product, models=product.models[:1]))
    original = models.load_model

    def run(points, *, as_of=ISSUE):
        inputs = ForecastInputs(as_of=as_of, read_at=as_of, observations=Observation(points=[]),
                                f107_measurements=points)
        result, = calculation.calculate_product(calculation.CalculationRequest('solar-radiation', inputs, ROOT, registry))
        assert ForecastArtifact(name=result.name, csv_text=result.content.decode(), sha256=sha256(result.content).hexdigest(),
                                row_count=result.row_count, columns=result.columns, model_info=result.model_info).issue_time() == ISSUE
        return result

    return run, registry, legacy_service, original


def test_daily_forecast_uses_the_exact_latest_receipt(runtime, monkeypatch):
    run, registry, _, original = runtime

    def only_daily(service, **kwargs):
        assert service is DailyF107Forecaster
        return original(service, **kwargs)

    monkeypatch.setattr(models, 'load_model', only_daily)
    late = ISSUE+timedelta(minutes=2)
    later = ISSUE+timedelta(minutes=11)
    measurements = [point(1, 115), point(3, 100), point(0, 120, receipt=late),
                    point(0, 999, receipt=later, source='celestrak.f107')]
    before = run(measurements)
    after = run(measurements, as_of=ISSUE+timedelta(minutes=10))
    early = pd.read_csv(BytesIO(before.content), parse_dates=['valid_time'])
    frame = pd.read_csv(BytesIO(after.content), parse_dates=['valid_time'])
    assert before.model_info['last_observation_at'] == point(1, 115).observed_at.isoformat()
    assert after.model_info['last_observation_at'] == point(0, 120).observed_at.isoformat()
    assert after.model_info['last_observation_received_at'] == late.isoformat()
    assert after.model_info['last_observation_source'] == 'swpc.f107'
    assert after.model_info['model'] == 'argus-f10-7-daily-v1'
    assert frame.lead_hours.tolist() == list(range(1, 121))
    assert early.loc[0, 'f107_q50'] == pytest.approx(115)
    assert frame.loc[0, ['f107_q10', 'f107_q50', 'f107_q90']].tolist() == [120.] * 3
    params = only_daily(DailyF107Forecaster, workdir=ROOT, registry=registry)[0].coefficients.days[1]
    expected = 120 + params.intercept + params.weights['trend3'] * 20
    assert frame.loc[1, 'f107_q50'] == pytest.approx(expected)
    assert frame.groupby(frame.valid_time.dt.date).f107_q50.nunique().eq(1).all()
    assert frame.f107_q10.le(frame.f107_q50).all() and frame.f107_q50.le(frame.f107_q90).all()


def test_three_day_history_gap_covers_the_full_horizon(runtime):
    run, _, _, _ = runtime
    artifact = run([point(3, 120)])
    frame = pd.read_csv(BytesIO(artifact.content), parse_dates=['valid_time'])
    assert artifact.model_info['model'] == 'argus-f10-7-daily-v1'
    assert frame.iloc[-1].valid_time == ISSUE + timedelta(hours=120)
    assert frame.f107_q50.eq(120).all()


@pytest.mark.parametrize('measurements', [[], [point(4, 120)], [point(0, 120, receipt=ISSUE+timedelta(minutes=1))]])
def test_missing_available_daily_history_keeps_the_legacy_forecast(runtime, monkeypatch, measurements):
    run, registry, legacy_service, original = runtime

    class LegacyForecast:
        registry_name = ARTIFACT

        def forecast_snapshot(self, inputs, *, issue_time):
            return pd.DataFrame([dict(issue_time=issue_time, lead_hours=1, valid_time=issue_time+timedelta(hours=1),
                                      f107_q10=98, f107_q50=100, f107_q90=105)])

    def load(service, **kwargs):
        if service is DailyF107Forecaster:
            return original(service, **kwargs)
        assert service is legacy_service
        assert kwargs['registry'] == {ARTIFACT: registry[ARTIFACT]['fallback']}
        return LegacyForecast(), {'model': 'argus-f10_7-q-v1', 'sha256': 'legacy'}

    monkeypatch.setattr(models, 'load_model', load)
    artifact = run(measurements)
    assert artifact.model_info['model'] == 'argus-f10_7-q-v1'
    assert artifact.model_info['fallback_for'] == 'argus-f10-7-daily-v1'
    assert pd.read_csv(BytesIO(artifact.content)).f107_q50.tolist() == [100]
