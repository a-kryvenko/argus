from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from argus_prophet import main
from argus_prophet.services import verification


def row(artifact, pairs, digest='model-a'):
    return {'artifact': artifact, 'evaluated_at': datetime(2026, 10, 3, tzinfo=UTC),
            'report': {'model_info': {'sha256': digest}, 'pairs': pairs}}


def pair(value, median, *, time='2026-09-10T00:00Z', lead=3, state='verified'):
    return {'valid_time': time, 'lead_hours': lead, 'value': value, 'state': state,
            'dst_q10': median-10, 'dst_q50': median, 'dst_q90': median+10}


def install_rows(monkeypatch, rows):
    cursor = Mock()
    cursor.fetchall.return_value = rows
    conn = Mock()
    conn.cursor.return_value = nullcontext(cursor)
    monkeypatch.setattr(verification, 'connect', lambda: nullcontext(conn))


def test_monthly_pools_errors_filters_valid_times_and_separates_models(monkeypatch):
    install_rows(monkeypatch, [
        row('dst_quantile', [pair(0, 0), pair(0, 4, lead=48),
            pair(None, 100, state='missing'), pair(None, 100, lead=12, state='pending'),
            pair(0, 100, time='2026-08-01T00:00Z'),
            pair(0, 100, time='2026-10-04T00:00Z')]),
        row('dst_quantile', [pair(0, 2)]),
        row('dst_quantile', [pair(0, 10)], 'model-b'),
    ])
    result = verification.monthly_accuracy('dst', now=datetime(2026, 10, 3, tzinfo=UTC))
    a, b = result.groups
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
    assert b.by_lead_hour[0].continuous['mae'] == 10
    assert result.start == datetime(2026, 9, 3, tzinfo=UTC)


def test_monthly_brier_and_empty_history(monkeypatch):
    pairs = [{'valid_time': '2026-09-10T00:00Z', 'lead_hours': lead,
              'value': value, 'state': 'verified',
              'p_kp_ge_4': probability, 'p_kp_ge_5': probability, 'p_kp_ge_6': probability}
             for lead, value, probability in [(3, 7, .8), (48, 0, .3)]]
    install_rows(monkeypatch, [row('kp_threshold', pairs)])
    result = verification.monthly_accuracy('geomagnetic-activity', now=datetime(2026, 10, 3, tzinfo=UTC))
    short, long = result.groups[0].by_lead_hour
    assert short.lead_hours == 3 and long.lead_hours == 48
    assert short.binary['4']['brier'] == pytest.approx(.04)
    assert long.binary['4']['brier'] == pytest.approx(.09)
    assert short.continuous is None
    install_rows(monkeypatch, [])
    assert verification.monthly_accuracy('dst').groups == []


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
