from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from app.services import density_history as service


def test_gfz_uses_observed_flux_and_rejects_missing_values():
    def row(day, observed):
        return ' '.join(map(str, [2026, 9, day, *([0]*22), observed, 999, 1]))
    frame = service.parse_gfz_f107('# GFZ daily observations\n' + row(1, 123.4) + '\n' + row(2, -1))
    assert frame.value.tolist() == [123.4]
    assert frame.observed_at.tolist() == [pd.Timestamp('2026-09-01T12:00Z')]
    with pytest.raises(ValueError, match='28 columns'):
        service.parse_gfz_f107('2026 09 01 1 2 3')


def test_merge_prefers_application_for_whole_solar_day():
    history = pd.DataFrame([
        dict(metric='s10', value=100, observed_at='2026-09-01T12:00Z'),
        dict(metric='s10', value=101, observed_at='2026-09-02T12:00Z'),
    ])
    app = pd.DataFrame([dict(metric='s10', value=99, observed_at='2026-09-02T10:00Z')])
    merged = service.merge_history(history, app)
    assert merged.value.tolist() == [100, 99]


def test_private_cache_reused_daily_and_invalidated_by_calibration(tmp_path, monkeypatch):
    calibration = tmp_path / 'calibration.json'
    calibration.write_text('version-one')
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(
        workdir=tmp_path, models_registry={'models': {
            'solar_index_calibration': {'calibration_path': calibration.name}}}))
    source = pd.DataFrame([dict(metric='f10_7', value=100., observed_at=pd.Timestamp('2026-09-05T12:00Z'))])
    download = Mock(return_value=source)
    monkeypatch.setattr(service, '_download_history', download)
    first = service.load_density_history(datetime(2026, 9, 6, 12, tzinfo=UTC))
    second = service.load_density_history(datetime(2026, 9, 6, 13, tzinfo=UTC))
    assert first.value.tolist() == second.value.tolist() == [100]
    assert download.call_count == 1
    calibration.write_text('version-two')
    service.load_density_history(datetime(2026, 9, 6, 14, tzinfo=UTC))
    assert download.call_count == 2
    assert len(list((tmp_path/'data/observations/jb2008').glob('*.parquet'))) == 2


def test_failed_refresh_preserves_cache(tmp_path, monkeypatch):
    calibration = tmp_path / 'calibration.json'
    calibration.write_text('version-one')
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(
        workdir=tmp_path, models_registry={'models': {
            'solar_index_calibration': {'calibration_path': calibration.name}}}))
    source = pd.DataFrame([dict(metric='s10', value=100., observed_at=pd.Timestamp('2026-09-05T12:00Z'))])
    monkeypatch.setattr(service, '_download_history', Mock(return_value=source))
    service.load_density_history(datetime(2026, 9, 6, 12, tzinfo=UTC))
    monkeypatch.setattr(service, '_download_history', Mock(side_effect=service.requests.RequestException('offline')))
    result = service.load_density_history(datetime(2026, 9, 7, 12, tzinfo=UTC))
    assert result.value.tolist() == [100]
