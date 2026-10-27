from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from prophet import main
from prophet.services import verification


def row(artifact, pairs, digest='model-a', model='test-model-v1', issue='2026-09-09T21:00:00+00:00'):
    return {'artifact': artifact, 'evaluated_at': datetime(2026, 10, 3, tzinfo=UTC),
            'issue_time': datetime.fromisoformat(issue),
            'report': {'model_info': {'model': model, 'sha256': digest}, 'pairs': pairs}}


def pair(value, median, *, time='2026-09-10T00:00Z', lead=3, state='verified'):
    return {'valid_time': time, 'lead_hours': lead, 'value': value, 'state': state,
            'dst_q10': median-10, 'dst_q50': median, 'dst_q90': median+10}


def install_rows(monkeypatch, rows):
    result = Mock()
    result.mappings.return_value.all.return_value = rows
    conn = Mock(execute=AsyncMock(return_value=result))
    conn.begin.return_value = nullcontext()
    monkeypatch.setattr(verification, 'get_db_session', lambda: nullcontext(conn))


@pytest.fixture(autouse=True)
def clear_monthly_cache():
    verification._monthly_cache.clear()
    yield
    verification._monthly_cache.clear()


async def test_monthly_cache_reuses_results_until_expiry_and_isolates_products(monkeypatch):
    clock = Mock(return_value=0)
    calculate = AsyncMock(side_effect=lambda product: object())
    monkeypatch.setattr(verification, 'monotonic', clock)
    monkeypatch.setattr(verification, 'monthly_accuracy', calculate)
    first = (await verification.cached_monthly_accuracy('dst'))
    clock.return_value = 299
    assert (await verification.cached_monthly_accuracy('dst')) is first
    assert (await verification.cached_monthly_accuracy('solar-wind-density')) is not first
    assert calculate.call_count == 2
    clock.return_value = 300
    assert (await verification.cached_monthly_accuracy('dst')) is not first
    assert calculate.call_count == 3


async def test_monthly_cache_does_not_cache_errors(monkeypatch):
    calculate = AsyncMock(side_effect=[RuntimeError('unavailable'), object()])
    monkeypatch.setattr(verification, 'monthly_accuracy', calculate)
    with pytest.raises(RuntimeError):
        (await verification.cached_monthly_accuracy('dst'))
    assert (await verification.cached_monthly_accuracy('dst')) is not None
    assert calculate.call_count == 2


async def test_monthly_separates_model_hashes_and_filters_valid_times(monkeypatch):
    install_rows(monkeypatch, [
        row('dst_quantile', [pair(0, 0), pair(0, 4, lead=48),
            pair(None, 100, state='missing'), pair(None, 100, lead=12, state='pending'),
            pair(0, 100, time='2026-08-01T00:00Z'),
            pair(0, 100, time='2026-10-04T00:00Z')]),
        row('dst_quantile', [pair(0, 2)], issue='2026-09-09T22:00:00+00:00'),
        row('dst_quantile', [pair(0, 10)], 'model-b'),
        row('dst_quantile', [pair(0, 100, time='2026-08-01T00:00Z')], issue='2026-10-02T00:00:00+00:00'),
        row('dst_quantile', []),
    ])
    result = (await verification.monthly_accuracy('dst', now=datetime(2026, 10, 3, tzinfo=UTC)))
    assert len(result.groups) == 2
    a = result.groups[0]
    assert a.model == 'test-model-v1'
    assert a.model_sha256 == 'model-a'
    assert a.last_issue_time == datetime(2026, 9, 9, 22, tzinfo=UTC)
    assert a.counts == {'verified': 3, 'missing': 1, 'pending': 1, 'total': 5}
    leads = {point.lead_hours: point for point in a.by_lead_hour}
    assert leads[3].continuous['mae'] == 1
    assert leads[3].continuous['rmse'] == pytest.approx(2**.5)
    assert leads[3].counts == {'verified': 2, 'missing': 1, 'pending': 0, 'total': 3}
    assert leads[48].continuous['mae'] == 4
    assert leads[48].continuous['rmse'] == 4
    assert leads[12].continuous is None
    assert leads[12].counts['pending'] == 1
    assert 96 not in leads
    assert a.releases == 2
    b = result.groups[1]
    assert b.model_sha256 == 'model-b'
    assert b.releases == 1
    assert b.counts['verified'] == 1
    assert b.by_lead_hour[0].continuous['mae'] == 10
    assert result.start == datetime(2026, 9, 3, tzinfo=UTC)


async def test_monthly_brier_and_empty_history(monkeypatch):
    pairs = [{'valid_time': '2026-09-10T00:00Z', 'lead_hours': lead,
              'value': value, 'state': 'verified',
              'p_kp_ge_4': probability, 'p_kp_ge_5': probability, 'p_kp_ge_6': probability}
             for lead, value, probability in [(3, 7, .8), (48, 0, .3)]]
    install_rows(monkeypatch, [row('kp_threshold', pairs),
        row('kp_threshold', [{**pairs[0], 'p_kp_ge_4': .1}], 'model-b')])
    result = (await verification.monthly_accuracy('geomagnetic-activity', now=datetime(2026, 10, 3, tzinfo=UTC)))
    assert len(result.groups) == 2
    assert result.groups[1].by_lead_hour[0].binary['4']['brier'] == pytest.approx(.81)
    short, long = result.groups[0].by_lead_hour
    assert short.lead_hours == 3 and long.lead_hours == 48
    assert short.binary['4']['brier'] == pytest.approx(.04)
    assert long.binary['4']['brier'] == pytest.approx(.09)
    assert short.continuous is None
    install_rows(monkeypatch, [])
    assert (await verification.monthly_accuracy('dst')).groups == []


def test_monthly_endpoint_requires_auth(monkeypatch):
    install_rows(monkeypatch, [])
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-secret')
    with TestClient(main.app) as client:
        path = '/internal/v1/forecasts/dst/verification'
        assert client.get(path).status_code == 401
        headers = {'Authorization': 'Bearer test-secret'}
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert response.json()['groups'] == []
        assert client.get('/internal/v1/forecasts/unknown/verification', headers=headers).status_code == 404


async def test_monthly_separates_names_and_missing_version_metadata(monkeypatch):
    unknown = row('dst_quantile', [pair(0, 30)])
    unknown['report'].pop('model_info')
    install_rows(monkeypatch, [
        row('dst_quantile', [pair(0, 1)], 'same-hash', 'model-v1'),
        row('dst_quantile', [pair(0, 2)], 'same-hash', 'model-v2'),
        row('dst_quantile', [pair(0, 3)], None, 'model-v2'),
        unknown,
        row('dst_quantile', [pair(0, 100, time='2026-08-01T00:00Z')], 'outside'),
    ])
    result = (await verification.monthly_accuracy('dst', now=datetime(2026, 10, 3, tzinfo=UTC)))
    assert [(g.model, g.model_sha256) for g in result.groups] == [
        ('model-v1', 'same-hash'), ('model-v2', 'same-hash'), ('model-v2', None), (None, None)]
    assert [g.by_lead_hour[0].continuous['mae'] for g in result.groups] == [1, 2, 3, 30]


pytestmark = pytest.mark.anyio
