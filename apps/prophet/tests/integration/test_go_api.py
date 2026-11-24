"""Exercise the Go HTTP API against published Prophet data."""
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import json
import os
import selectors
import shutil
import subprocess
from time import monotonic
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from common import config as common_config
from common.schemas.forecast_release import ForecastRelease, PRODUCT_ARTIFACTS
from common.schemas.forecast_status import ForecastsStatus
from common.schemas.forecast_verification import ForecastVerification
from common.schemas.observation import Observation
from common.schemas.solar_indices import SolarIndices
from forecast.inputs import ForecastInputs
from prophet.db.locks import transaction
from prophet.services import verification
from prophet.services.releases.publication import read_release
from prophet.services.releases.recording import RunRecorder
from tests.support.database import ROOT
from .storage import recorder_database, recorder_setup, store_product

pytestmark = pytest.mark.anyio


@contextmanager
def go_server(workdir):
    if shutil.which('go') is None:
        pytest.skip('Go is required')
    binary = workdir / 'prophet-api'
    subprocess.run(
        ['go', 'build', '-o', str(binary), '.'],
        cwd=ROOT / 'apps/prophet/api-go', check=True, timeout=120,
    )
    environment = {**os.environ, 'ARGUS_WORKDIR': str(workdir), 'PROPHET_HTTP_ADDR': '127.0.0.1:0'}
    with subprocess.Popen([binary], env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=10), 'Prophet Go API did not start'
                line = process.stdout.readline()
            _, marker, address = line.partition('Prophet listening on ')
            assert marker, line
            yield binary, f'http://{address.strip()}'
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


async def test_go_read_contract_and_prepared_rolling_verification(recorder_setup, monkeypatch):
    _, _, config, writer = recorder_setup
    workdir = config.workdir
    now = datetime.now(UTC)
    issue = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=4)
    monkeypatch.setenv('ARGUS_WORKDIR', str(workdir))
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-token')
    monkeypatch.setattr(common_config, '_config', None)

    public_summary = {'generated_at': now.isoformat(), 'observations': {'test': 'сводка'}}
    summary_file = workdir / 'data/prophet/summary/latest.json'
    summary_file.parent.mkdir(parents=True)
    for product in ('dst', 'solar-wind-speed', 'solar-wind-events'):
        run = await RunRecorder.begin(product, 'manual', config, writer=writer)
        await store_product(run, names=PRODUCT_ARTIFACTS[product], issue=issue.isoformat())
        result = await run.finish()
        if product == 'dst':
            historical = result.release_id
    run = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await run.snapshot(ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[])))
    await store_product(run, issue=(issue + timedelta(hours=1)).isoformat())
    current = await run.finish()
    failure = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await failure.finish(error=ValueError('model unavailable'))

    current_speed = await read_release('solar-wind-speed')
    stale_speed = await RunRecorder.begin('solar-wind-speed', 'manual', config, writer=writer)
    await store_product(stale_speed, names=PRODUCT_ARTIFACTS['solar-wind-speed'],
                        issue=(issue - timedelta(hours=1)).isoformat())
    kept_current = await stale_speed.finish()
    assert kept_current.status == 'kept_current'

    solar = await RunRecorder.begin('solar-radiation', 'manual', config, writer=writer)
    await solar.snapshot(ForecastInputs(
        as_of=now, read_at=now, observations=Observation(points=[]),
        measurements=[
            {'metric': 's10', 'value': 110, 'observed_at': now},
            {'metric': 's10', 'value': 111, 'observed_at': now - timedelta(days=15)},
            {'metric': 'dst', 'value': -10, 'observed_at': now},
        ],
    ))
    async with transaction(writer) as session:
        await session.execute(text("UPDATE forecast_run SET product='wind' WHERE id=:run_id"),
                              {'run_id': stale_speed.run_id})
        await session.execute(text("""INSERT INTO forecast_verification(release_id,artifact,evaluated_at,report)
            VALUES (:release_id,'dst_quantile',:now,:report)""").bindparams(bindparam('report', type_=JSONB)),
            {'release_id': current.release_id, 'now': now, 'report': {
                'model_info': {'model': 'test-model', 'sha256': 'test-hash'},
                'pairs': [{'valid_time': (issue + timedelta(hours=2)).isoformat(),
                           'lead_hours': 1, 'state': 'verified', 'value': 0,
                           'dst_q10': -1, 'dst_q50': 2, 'dst_q90': 3}],
            }})

    with go_server(workdir) as (binary, address):
        async with httpx.AsyncClient(base_url=address, trust_env=False) as go:
            headers = {'Authorization': 'Bearer test-token'}
            prefix = '/internal/v1/forecasts/'
            verification_path = prefix + 'dst/verification'
            summary_path = '/internal/v1/summary'
            assert (await go.get('/health/ready')).json() == {'service': 'prophet', 'status': 'ok'}
            for protected in (prefix + 'dst/latest', prefix + 'status', verification_path,
                              summary_path, '/internal/v1/solar-indices'):
                assert (await go.get(protected)).status_code == 401
            assert (await go.get(verification_path, headers=headers)).status_code == 503
            assert (await go.get(summary_path, headers=headers)).status_code == 503
            summary_file.write_text(json.dumps(public_summary, ensure_ascii=False), encoding='utf-8')
            assert (await go.get(summary_path, headers=headers)).json() == public_summary

            for suffix, code in [
                ('unknown/latest', 404), ('hmf/latest', 503), (f'dst/releases/{uuid4()}', 404),
                ('dst/releases/invalid', 422), ('unknown/verification', 404), ('solar-wind-events/verification', 404),
            ]:
                assert (await go.get(prefix + suffix, headers=headers)).status_code == code

            for product, release_id in [('dst', None), ('dst', historical),
                                        ('solar-wind-speed', None), ('solar-wind-events', None)]:
                suffix = f'{product}/releases/{release_id}' if release_id else f'{product}/latest'
                response = await go.get(prefix + suffix, headers=headers)
                assert ForecastRelease.model_validate(response.json()) == await read_release(product, release_id)

            await verification.publish_monthly_accuracy(now=now, deadline=monotonic() + 30)
            await verification.publish_monthly_accuracy(now=now - timedelta(minutes=1), deadline=monotonic() + 30)
            report = ForecastVerification.model_validate((await go.get(verification_path, headers=headers)).json())
            assert report.end == now
            assert report.groups[0].by_lead_hour[0].continuous['mae'] == 2
            empty = ForecastVerification.model_validate(
                (await go.get(prefix + 'hmf/verification', headers=headers)).json(),
            )
            assert empty.groups == []

            path = '/internal/v1/solar-indices'
            actual = SolarIndices.model_validate((await go.get(path, headers=headers)).json())
            assert actual.as_of == now
            assert [(item.metric, item.value, item.observed_at) for item in actual.items] == [('s10', 110, now)]

            actual = ForecastsStatus.model_validate((await go.get(prefix + 'status', headers=headers)).json())
            assert [item.product for item in actual.products] == list(PRODUCT_ARTIFACTS)
            statuses = {item.product: item for item in actual.products}
            dst = statuses['dst']
            assert dst.current_release.release_id == current.release_id
            assert dst.current_release.input_diagnostics['normalized']['count'] == 0
            assert dst.freshness == 'unconfigured'
            assert dst.latest_attempt.run_id == failure.run_id
            assert dst.latest_attempt.status == 'failed'
            assert dst.latest_attempt_artifacts == []
            speed = statuses['solar-wind-speed']
            assert speed.current_release.release_id == current_speed.release_id
            assert speed.latest_attempt.run_id == stale_speed.run_id
            assert speed.latest_attempt.publication == kept_current
            assert [item.name for item in speed.latest_attempt_artifacts] == list(PRODUCT_ARTIFACTS['solar-wind-speed'])
            assert statuses['hmf'].freshness == 'unavailable'
            assert statuses['hmf'].current_release is None
            assert statuses['hmf'].latest_attempt is None
            assert statuses['hmf'].latest_attempt_artifacts == []

            result = await asyncio_smoke(binary, address)
            assert result.returncode == 0, result.stderr


async def asyncio_smoke(binary, address):
    import asyncio

    return await asyncio.to_thread(
        subprocess.run, [binary, 'smoke', address, ''], capture_output=True, text=True, timeout=20,
    )


async def test_public_forecasts_match_existing_contract_and_refresh(recorder_setup, monkeypatch):
    import asyncio
    import csv
    from io import StringIO
    from tests.support.clio import go_http_server

    _, _, config, writer = recorder_setup
    issue = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    monkeypatch.setenv('ARGUS_WORKDIR', str(config.workdir))
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-token')
    monkeypatch.setattr(common_config, '_config', None)
    dst = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await store_product(dst, issue=issue.isoformat())
    await dst.finish()

    density = await RunRecorder.begin('atmospheric-density', 'manual', config, writer=writer)
    rows = [dict(issue_time=issue.isoformat(), valid_time=(issue + timedelta(hours=lead)).isoformat(),
                 lead_hours=lead, observed_at=(issue-timedelta(days=5)).isoformat(),
                 driver_mode='observed_persistence', background_method='trailing_81_daily_values',
                 dtc_method='causal_dst_ap_v1', history_start=(issue-timedelta(days=85)).isoformat(),
                 dtc_observed_at=issue.isoformat(), altitude_km=400, latitude_deg=0,
                 rho_kg_m3=1e-12, rho_lon_p10_kg_m3=8e-13, rho_lon_p90_kg_m3=2e-12)
            for lead in range(49)]
    stream = StringIO()
    csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    csv_writer.writeheader()
    csv_writer.writerows(rows)
    await density.store('atmospheric_density', stream.getvalue().encode(), {}, len(rows), list(rows[0]))
    await density.finish()

    events = await RunRecorder.begin('solar-wind-events', 'manual', config, writer=writer)
    event_csv = await store_product(events, names=PRODUCT_ARTIFACTS['solar-wind-events'], issue=issue.isoformat())
    await events.finish()
    event_report = json.loads(next(csv.DictReader(StringIO(event_csv)))['report_json'])

    model_registry = {'models': {'dst_quantile': {'metrics_report': 'dst-metrics.json'}}}
    (config.workdir / 'configs/models_registry.yaml').write_text(json.dumps(model_registry))
    (config.workdir / 'dst-metrics.json').write_text(json.dumps({
        'target': 'dst',
        'variables': {'dst': {'continuous': {
            'quantiles': [0.1, 0.5, 0.9],
            'by_lead_hour': [
                {'lead_hours': 48, 'values': {'mae': 2.0}},
                {'lead_hours': 49, 'values': {'mae': 3.0}},
            ],
        }}},
    }))
    targets = ('dst', 'atmospheric-density')
    releases = [(await read_release(target)).model_dump(mode='json') for target in targets]
    reference = await asyncio.to_thread(reference_projections, releases, config.workdir)
    with go_server(config.workdir) as (_, prophet_url):
        environment = {**os.environ, 'FORECASTS_URL': prophet_url}
        with go_http_server(config.workdir, environment, ROOT / 'apps/api/api-go', 'public-api',
                            'PUBLIC_API_HTTP_ADDR', 'Public API listening on ') as public_url:
            async with httpx.AsyncClient(base_url=public_url, trust_env=False) as public, \
                       httpx.AsyncClient(base_url=prophet_url, trust_env=False) as prophet:
                assert (await prophet.get('/internal/v1/public-forecasts/dst')).status_code == 401
                assert (await prophet.get('/internal/v1/public-forecast-metrics/dst')).status_code == 401

                async def read_forecast(target, meta, accept=lambda _: True):
                    deadline = monotonic() + 8
                    while True:
                        response = await public.get(f'/api/v1/forecasts/{target}', params={'meta': str(meta).lower()})
                        if response.status_code == 200:
                            data = response.json()['data']
                            if accept(data):
                                return data
                        assert monotonic() < deadline, response.text
                        await asyncio.sleep(0.05)

                for target in targets:
                    for meta in (False, True):
                        assert await read_forecast(target, meta) == reference[target][int(meta)]
                assert await read_forecast('solar-wind-events', False) == event_report
                for meta in (False, True):
                    metrics = await public.get('/api/v1/forecasts/dst/metrics', params={'meta': str(meta).lower()})
                    assert metrics.status_code == 200
                    assert metrics.json()['data'] == reference['dst-metrics'][int(meta)]
                assert (await public.get('/api/v1/forecasts/hmf/metrics')).status_code == 503

                updated_issue = issue + timedelta(hours=1)
                updated = await RunRecorder.begin('dst', 'manual', config, writer=writer)
                await store_product(updated, issue=updated_issue.isoformat())
                await updated.finish()
                received = await read_forecast('dst', False, lambda data: datetime.fromisoformat(data['issue_time']) == updated_issue)
                assert received['predictions'][0]['valid_time'] == (updated_issue + timedelta(hours=1)).isoformat().replace('+00:00', 'Z')


def reference_projections(releases, workdir):
    code = """
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from apps.api.tests.reference_public.services.forecasts import products
from apps.api.tests.reference_public.schemas.atmospheric_density import DensityForecast

releases = {item['product']: item for item in json.load(sys.stdin)}
workdir = Path(sys.argv[1])
registry = json.loads((workdir / 'configs/models_registry.yaml').read_text())
registry['models']['atmospheric_density'] = {'max_age_hours': 6}
products.get_config = lambda: SimpleNamespace(workdir=workdir, models_registry=registry)
products.read_frames = lambda target: {
    artifact['name']: pd.read_csv(io.StringIO(artifact['csv_text']), parse_dates=['issue_time', 'valid_time'])
    for artifact in releases[target]['artifacts']
}
print(json.dumps({
    'dst': [products.load_forecast(products.PRODUCTS['dst'], meta=meta).model_dump(mode='json')
            for meta in (False, True)],
    'atmospheric-density': [DensityForecast.from_internal(products.load_density_forecast(), meta=meta).model_dump(mode='json')
                            for meta in (False, True)],
    'dst-metrics': [products.load_metrics(products.PRODUCTS['dst'], meta=meta).model_dump(mode='json')
                    for meta in (False, True)],
}))
"""
    result = subprocess.run([ROOT / 'apps/api/.venv/bin/python', '-c', code, str(workdir)], cwd=ROOT,
                            input=json.dumps(releases), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
