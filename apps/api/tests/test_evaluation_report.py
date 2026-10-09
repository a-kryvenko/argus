from pathlib import Path
from types import SimpleNamespace

import pytest
from app.services import forecast_products as service
from app.services.forecast_errors import ArtifactNotReadyError

ROOT = Path(__file__).resolve().parents[3]


def test_new_blend_report_replaces_legacy_quantiles_and_probabilities(monkeypatch):
    import yaml
    registry = yaml.safe_load((ROOT/'configs/models_registry.yaml').read_text())
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(workdir=ROOT, models_registry=registry))
    monkeypatch.setattr(service, '_binary_metrics', lambda *_: pytest.fail('Legacy probabilities must not be loaded'))
    result = service.load_metrics(service.PRODUCTS['solar-wind-speed'])
    assert '2026' in result.evaluation.period
    wind = result.variables['v']
    assert wind.binary == [] and wind.continuous.quantiles == []
    assert len(wind.continuous.by_lead_hour) == 96
    last = wind.continuous.by_lead_hour[-1]
    assert last.values['n'] == 156
    assert last.values['mae'] == pytest.approx(56.33067077414393, abs=.00001)
    assert 'coverage_80' not in last.values


def test_missing_new_report_never_falls_back_to_old_metrics(tmp_path, monkeypatch):
    registry = {'models': {'plasma_speed_quantile': {'metrics_report': 'missing.json'}}}
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(workdir=tmp_path, models_registry=registry))
    with pytest.raises(ArtifactNotReadyError):
        service.load_metrics(service.PRODUCTS['solar-wind-speed'])


def test_density_report_uses_blend_intervals_and_matched_nrt_cohort(monkeypatch):
    import yaml
    registry = yaml.safe_load((ROOT/'configs/models_registry.yaml').read_text())
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(workdir=ROOT, models_registry=registry))
    result = service.load_metrics(service.PRODUCTS['solar-wind-density'])
    assert '156 daily' in result.evaluation.sample
    density = result.variables['n'].continuous
    assert density.quantiles == [.1, .5, .9]
    assert len(density.by_lead_hour) == 96
    last = density.by_lead_hour[-1]
    assert last.values['n'] == 156
    assert last.values['mae'] == pytest.approx(2.7292544318690695)
    assert 'coverage_80' in last.values


def test_temperature_report_covers_all_hours_and_calibrated_intervals(monkeypatch):
    import yaml
    registry = yaml.safe_load((ROOT/'configs/models_registry.yaml').read_text())
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(workdir=ROOT, models_registry=registry))
    result = service.load_metrics(service.PRODUCTS['solar-wind-temperature'], meta=True)
    assert result.meta.variables['t'].unit == 'K'
    temperature = result.variables['t'].continuous
    assert temperature.quantiles == [.1, .5, .9]
    assert [row.lead_hours for row in temperature.by_lead_hour] == list(range(1, 97))
    assert temperature.by_lead_hour[-1].values['mae'] == pytest.approx(51892.1, abs=.1)
    assert 0 <= temperature.by_lead_hour[-1].values['coverage_80'] <= 1
