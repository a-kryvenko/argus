import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from common.schemas.forecast_release import ForecastArtifact, ForecastRelease
from argus_prophet import main
from argus_prophet.publication import ReleaseNotFound


def artifact(name='dst_quantile', value='-10'):
    text = 'issue_time,valid_time,lead_hours,dst_q10,dst_q50,dst_q90\n2026-09-13T00:00:00Z,2026-09-13T01:00:00Z,1,-20,' + value + ',0\n'
    return ForecastArtifact(name=name, sha256=hashlib.sha256(text.encode()).hexdigest(),
                            csv_text=text, row_count=1, columns=text.splitlines()[0].split(','), model_info={})


def release():
    return ForecastRelease(release_id=uuid4(), run_id=uuid4(), product='dst',
                           published_at=datetime.now(UTC), issue_time=datetime(2026, 9, 13, tzinfo=UTC),
                           artifacts=[artifact()])


def test_rejects_corrupt_and_incomplete_releases():
    payload = release().model_dump()
    payload['artifacts'][0]['csv_text'] += 'corrupt'
    with pytest.raises(ValueError, match='checksum'):
        ForecastRelease.model_validate(payload)
    payload = release().model_dump()
    payload['product'] = 'solar-wind-speed'
    with pytest.raises(ValueError, match='Incomplete'):
        ForecastRelease.model_validate(payload)
    payload = release().model_dump()
    payload['issue_time'] = datetime(2026, 9, 12, tzinfo=UTC)
    with pytest.raises(ValueError, match='Mixed'):
        ForecastRelease.model_validate(payload)


def test_internal_reads_require_auth_and_return_exact_release(monkeypatch):
    expected = release()
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-secret')
    calls = []
    def read(product, release_id=None):
        calls.append((product, release_id))
        return expected
    monkeypatch.setattr(main, 'read_release', read)
    with TestClient(main.app) as client:
        path = '/internal/v1/forecasts/dst/latest'
        assert client.get(path).status_code == 401
        assert client.get(path, headers={'Authorization': 'Bearer wrong'}).status_code == 401
        assert calls == []
        headers = {'Authorization': 'Bearer test-secret'}
        response = client.get(path, headers=headers)
        assert ForecastRelease.model_validate(response.json()) == expected
        assert client.get(f'/internal/v1/forecasts/dst/releases/{expected.release_id}', headers=headers).status_code == 200
        assert calls[-1] == ('dst', expected.release_id)
        assert client.get('/internal/v1/forecasts/unknown/latest', headers=headers).status_code == 404


def test_missing_release_storage_failure_and_missing_configuration(monkeypatch):
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-secret')
    with TestClient(main.app) as client:
        headers = {'Authorization': 'Bearer test-secret'}
        for failure in (ReleaseNotFound('dst'), RuntimeError('private database details')):
            def read(*_, **__):
                raise failure
            monkeypatch.setattr(main, 'read_release', read)
            response = client.get('/internal/v1/forecasts/dst/latest', headers=headers)
            assert response.status_code == 503
            assert 'private database' not in response.text
        monkeypatch.delenv('FORECASTS_SERVICE_TOKEN')
        assert client.get('/internal/v1/forecasts/dst/latest', headers=headers).status_code == 503


@pytest.mark.parametrize('value', ['nan', 'inf', 'not-a-number'])
def test_nonfinite_or_invalid_predictions_cannot_be_published(value):
    with pytest.raises(ValueError):
        artifact(value=value)


def test_missing_prediction_columns_are_rejected():
    payload = artifact().model_dump()
    text = 'issue_time,valid_time,lead_hours\n2026-09-13T00:00:00Z,2026-09-13T01:00:00Z,1\n'
    payload.update(csv_text=text, columns=text.splitlines()[0].split(','), sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError, match='prediction columns'):
        ForecastArtifact.model_validate(payload)


def test_status_contract_requires_auth_and_does_not_change_latest_reads(monkeypatch):
    from argus_prophet import readiness
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-secret')
    monkeypatch.setattr(readiness, 'product_status', lambda product: dict(product=product, assessed_at=datetime.now(UTC), current_release=None,
        release_age_hours=None, existing_public_max_age_hours=None, freshness='unavailable',
        latest_attempt=None, latest_attempt_artifacts=[]))
    with TestClient(main.app) as client:
        path = '/internal/v1/forecasts/dst/status'
        assert client.get(path).status_code == 401
        response = client.get(path, headers={'Authorization': 'Bearer test-secret'})
        assert response.status_code == 200 and response.json()['mode'] == 'observe'
