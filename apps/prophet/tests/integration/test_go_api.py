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
