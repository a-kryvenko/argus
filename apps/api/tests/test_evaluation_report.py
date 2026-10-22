import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from api.services.forecasts import products as service
from api.services.errors import ArtifactNotReadyError

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def report_config(tmp_path, monkeypatch):
    registry = {'models': {}}
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(workdir=tmp_path, models_registry=registry))

    def install(target, calibrated):
        product = service.PRODUCTS[target]
        variable = product.variables[0]
        rows = [dict(lead_hours=lead, values=dict(mae=12.5, n=20,
                    **({'coverage_80': .8} if calibrated else {})))
                for lead in (1, product.max_horizon_hours, product.max_horizon_hours + 1)]
        binary = [dict(threshold=450, by_lead_hour=[dict(lead_hours=row['lead_hours'], brier_score=0.)
                  for row in rows])] if calibrated else []
        report = dict(target=target, evaluation=dict(label='Synthetic evaluation', period='Test period',
                      sample='Test cohort', notes=['Synthetic data']), variables={variable.name: dict(
                      continuous=dict(quantiles=[.1, .5, .9] if calibrated else [], by_lead_hour=rows), binary=binary)})
        path = tmp_path / 'report.json'
        path.write_text(json.dumps(report))
        registry['models'][variable.registry_name] = dict(model='test-model', metrics_report='report.json')
        return product, variable

    return install


@pytest.mark.parametrize('calibrated', [False, True], ids=['point-only', 'calibrated'])
@pytest.mark.parametrize('target', ['solar-wind-speed', 'solar-wind-density', 'solar-wind-temperature'])
def test_report_limits_horizon(report_config, target, calibrated):
    product, variable = report_config(target, calibrated)
    result = service.load_metrics(product, meta=True)
    assert result.evaluation.sample == 'Test cohort'
    assert result.meta.variables[variable.name].unit == variable.unit
    metrics = result.variables[variable.name]
    assert metrics.continuous.quantiles == ([.1, .5, .9] if calibrated else [])
    assert [row.lead_hours for row in metrics.continuous.by_lead_hour] == [1, product.max_horizon_hours]
    assert all(row.values['mae'] == 12.5 and row.values['n'] == 20 for row in metrics.continuous.by_lead_hour)
    if calibrated:
        assert all(row.values['coverage_80'] == .8 for row in metrics.continuous.by_lead_hour)
        assert len(metrics.binary) == 1 and metrics.binary[0].threshold == 450
        assert [row.lead_hours for row in metrics.binary[0].by_lead_hour] == [1, product.max_horizon_hours]
        assert all(row.brier_score == 0 for row in metrics.binary[0].by_lead_hour)
    else:
        assert metrics.binary == []
        assert all('coverage_80' not in row.values for row in metrics.continuous.by_lead_hour)


def test_missing_report_is_unavailable(report_config, tmp_path):
    product, _ = report_config('solar-wind-speed', True)
    (tmp_path / 'report.json').unlink()
    with pytest.raises(ArtifactNotReadyError):
        service.load_metrics(product)


def test_registered_reports_match_api_schema():
    import yaml
    registry = yaml.safe_load((ROOT / 'configs/models_registry.yaml').read_text())
    for product in service.PRODUCTS.values():
        paths = {registry['models'].get(variable.registry_name, {}).get('metrics_report')
                 for variable in product.variables} - {None}
        for path in paths:
            report = service.ForecastMetrics.model_validate_json((ROOT / path).read_text())
            assert report.target == product.target
            assert set(report.variables) <= {variable.name for variable in product.variables}
