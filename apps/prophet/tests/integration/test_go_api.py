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


def test_f107_history_ranks_only_measurements_received_at_cutoff(recorder_database):
    import psycopg
    from prophet.services.inputs.observations import read_f107_history
    from tests.support.clio import clio_server
    from tests.support.database import database_environment

    _, urls, config = recorder_database
    observed_at = datetime.now(UTC).replace(hour=20, minute=0, second=0, microsecond=0) - timedelta(days=2)
    first_receipt = observed_at + timedelta(minutes=15)
    later_receipt = observed_at + timedelta(hours=3)
    with psycopg.connect(urls['clio']) as db, db.cursor() as cursor:
        cursor.executemany('''INSERT INTO clio.measurement
            (metric, observed_at, value, source_product, received_at) VALUES ('f10_7', %s, %s, %s, %s)''', [
            (observed_at - timedelta(hours=3), 129., 'swpc.f107', observed_at - timedelta(hours=2)),
            (observed_at, 130., 'swpc.f107', first_receipt),
            (observed_at + timedelta(hours=2), 132., 'swpc.f107', observed_at + timedelta(hours=2, minutes=5)),
            (observed_at, 131., 'celestrak.f107', later_receipt),
        ])
    token = 'test-observations-token'
    environment = {**os.environ, **database_environment(urls), 'ARGUS_WORKDIR': str(config.workdir),
                   'OBSERVATIONS_SERVICE_TOKEN': token}
    with clio_server(config.workdir, environment) as clio_url, httpx.Client(trust_env=False) as client:
        before_noon = read_f107_history(client, clio_url, token, observed_at - timedelta(hours=1))
        earlier = read_f107_history(client, clio_url, token, first_receipt)
        afternoon = read_f107_history(client, clio_url, token, later_receipt - timedelta(minutes=30))
        later = read_f107_history(client, clio_url, token, later_receipt)

    assert before_noon == []
    assert [(item.observed_at, item.received_at, item.source_product, item.value) for item in earlier] == [
        (observed_at, first_receipt, 'swpc.f107', 130.),
    ]
    assert afternoon == earlier
    assert [(item.observed_at, item.received_at, item.source_product, item.value) for item in later] == [
        (observed_at, later_receipt, 'celestrak.f107', 131.),
    ]


async def test_ordinal_geomagnetic_release_from_clio_to_public_api(recorder_setup, monkeypatch):
    import psycopg
    import yaml
    from prophet.services.generation.calculation import CalculationRequest, calculate_product
    from prophet.services.inputs.loader import load_product_inputs
    from tests.support.clio import clio_server, go_http_server
    from tests.support.database import database_environment

    _, urls, config, writer = recorder_setup
    issue = (datetime.now(UTC) - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    as_of = issue + timedelta(minutes=10)
    kp_start = issue - timedelta(hours=issue.hour % 3 + 3)
    previous_day = issue.replace(hour=0) - timedelta(days=1)
    with psycopg.connect(urls['clio']) as db, db.cursor() as cursor:
        rows = [(metric, issue - timedelta(minutes=minute), value, source, issue)
                for minute in range(1, 181)
                for metric, value, source in (('n', 4., 'swpc.rtsw_plasma'), ('v', 504., 'swpc.rtsw_plasma'),
                    ('by', 2., 'swpc.rtsw_mag'), ('bz', -6., 'swpc.rtsw_mag'))]
        rows.extend([
            ('kp', kp_start, 4.67, 'swpc.kp', issue),
            ('dst', issue - timedelta(hours=1), -29., 'swpc.dst', issue),
            ('f10_7', previous_day, 130., 'swpc.f107', previous_day + timedelta(hours=20)),
        ])
        cursor.executemany('''INSERT INTO clio.measurement
            (metric, observed_at, value, source_product, received_at) VALUES (%s,%s,%s,%s,%s)''', rows)
        cursor.execute('''INSERT INTO clio.normalized_observation
            (observed_at,bx,by,bz,v,n,t,kp,dst,ap,f10_7,wind_time_basis)
            VALUES (%s,3,2,-8,530,5,120000,5.33,-22,56,130,'l1')''',
            (kp_start - timedelta(days=27),))

    monkeypatch.setenv('ARGUS_WORKDIR', str(config.workdir))
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-observations-token')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-forecasts-token')
    monkeypatch.setattr(common_config, '_config', None)
    environment = {**os.environ, **database_environment(urls)}

    def executor(target, *args, timeout_seconds, **kwargs):
        return target(*args, **kwargs)

    with clio_server(config.workdir, environment) as clio_url:
        monkeypatch.setenv('OBSERVATIONS_URL', clio_url)
        inputs = await load_product_inputs('geomagnetic-activity', as_of, {}, executor=executor, timeout_seconds=30)

    prior = {point.time: point for point in inputs.geomagnetic_history.prior}
    assert prior[issue].kp == 4.67
    assert prior[issue].dst == -29
    assert prior[issue].bz == -6
    assert any(point.kp == 5.33 for point in inputs.geomagnetic_history.raw)
    registry = yaml.safe_load((ROOT / 'configs/models_registry.yaml').read_text())['models']
    product = 'geomagnetic-activity'
    artifacts = calculate_product(CalculationRequest(product, inputs, ROOT, registry))
    assert {artifact.model_info['model'] for artifact in artifacts} == {'argus-geomagnetic-ordinal-v1'}
    assert len({artifact.model_info['sha256'] for artifact in artifacts}) == 1
    run = await RunRecorder.begin(product, 'manual', config, writer=writer)
    await run.snapshot(inputs)
    for artifact in artifacts:
        await run.store(artifact.name, artifact.content, artifact.model_info, artifact.row_count, artifact.columns)
    assert (await run.finish()).status == 'published'
    assert (await read_release(product)).issue_time == issue

    registry_path = ROOT / 'configs/models_registry.yaml'
    shutil.copyfile(registry_path, config.workdir / 'configs/models_registry.yaml')
    evaluation = registry['geomagnetic_ordinal']['metrics_report']
    (config.workdir / evaluation).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / evaluation, config.workdir / evaluation)

    with go_server(config.workdir) as (_, prophet_url), go_http_server(
            config.workdir, {**os.environ, 'FORECASTS_URL': prophet_url}, ROOT / 'apps/api/api-go', 'public-api',
            'PUBLIC_API_HTTP_ADDR', 'Public API listening on ') as public_url:
        async with httpx.AsyncClient(base_url=public_url, trust_env=False) as client, \
                   httpx.AsyncClient(base_url=prophet_url, trust_env=False) as prophet:
            response = await client.get(f'/api/v1/forecasts/{product}')
            metrics = await client.get(f'/api/v1/forecasts/{product}/metrics')
            status = await prophet.get('/internal/v1/forecasts/status',
                                       headers={'Authorization': 'Bearer test-forecasts-token'})
    metrics.raise_for_status()
    published = metrics.json()['data']
    assert published['target'] == product
    assert set(published['variables']) == {'kp', 'ap'}
    kp_metrics = published['variables']['kp']['binary']
    assert [item['threshold'] for item in kp_metrics] == [4, 5, 6]
    ap_metrics = published['variables']['ap']['continuous']
    for series in [*kp_metrics, ap_metrics]:
        assert [row['lead_hours'] for row in series['by_lead_hour']] == list(range(1, 49))
    status.raise_for_status()
    products = ForecastsStatus.model_validate(status.json()).products
    diagnostics = next(item.current_release.input_diagnostics for item in products if item.product == product)
    assert diagnostics['issues'] == []
    assert datetime.fromisoformat(diagnostics['geomagnetic_history']['latest_hour']) == issue
    assert datetime.fromisoformat(diagnostics['geomagnetic_history']['latest_kp_interval']) == kp_start
    response.raise_for_status()
    forecast = response.json()['data']
    assert set(forecast['available_variables']) == {'kp', 'ap'}
    assert forecast['horizon_hours'] == 48
    assert [row['lead_hours'] for row in forecast['predictions']] == list(range(1, 49))
    for row in forecast['predictions']:
        variables = row['variables']
        assert [item['threshold'] for item in variables['kp']['binary']] == [4, 5, 6]
        ap = variables['ap']['continuous']
        assert 0 <= ap['q10'] <= ap['q50'] <= ap['q90']


async def test_fast_l1_updates_one_prepared_interval_and_preserves_archived_release(recorder_setup, monkeypatch):
    import asyncio

    from common.schemas.forecast_release import FAST_L1_ARTIFACT, FAST_L1_COLUMNS, FAST_L1_PRODUCT, PREDICTION_COLUMNS

    _, _, config, writer = recorder_setup
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-token')
    monkeypatch.setattr(common_config, '_config', None)
    headers = {'Authorization': 'Bearer test-token'}
    product = 'geomagnetic-activity'
    issue = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    base = await RunRecorder.begin(product, 'manual', config, writer=writer)
    for name, values in [('kp_threshold', '.1,.05,.01'), ('ap_quantile', '2,6,16')]:
        columns = ['issue_time', 'valid_time', 'lead_hours', *PREDICTION_COLUMNS[name]]
        rows = [f'{issue.isoformat()},{(issue + timedelta(hours=lead)).isoformat()},{lead},{values}'
                for lead in range(1, 49)]
        await base.store(name, ('\n'.join([','.join(columns), *rows]) + '\n').encode(), {}, len(rows), columns)
    release = await base.finish()
    archive_path = f'/internal/v1/forecasts/{product}/releases/{release.release_id}'
    view_path = f'/internal/v1/public-forecasts/{product}'

    with go_server(config.workdir) as (_, address):
        async with httpx.AsyncClient(base_url=address, headers=headers, trust_env=False) as go:
            deadline = monotonic() + 8
            while True:
                response = await go.get(view_path)
                if response.status_code != 503 or monotonic() >= deadline:
                    before = response.raise_for_status().json()
                    break
                await asyncio.sleep(.05)
            archived = ForecastRelease.model_validate((await go.get(archive_path)).raise_for_status().json())
            assert archived.release_id == release.release_id
            assert len(before['predictions']) == 48

            fast_issue = datetime.now(UTC)
            target = (fast_issue.replace(minute=0, second=0, microsecond=0)
                      + timedelta(hours=3 - fast_issue.hour % 3))
            remaining = (target - fast_issue).total_seconds()
            if remaining < 10:
                await asyncio.sleep(remaining + .01)
                fast_issue = datetime.now(UTC)
                target += timedelta(hours=3)
            fast = await RunRecorder.begin(FAST_L1_PRODUCT, 'scheduled', config, writer=writer)
            columns = ['issue_time', 'valid_time', *FAST_L1_COLUMNS]
            content = f"{','.join(columns)}\n{fast_issue.isoformat()},{target.isoformat()},.8,.7,.4,9,32,80\n"
            await fast.store(FAST_L1_ARTIFACT, content.encode(), {}, 1, columns)
            assert (await fast.finish()).status == 'published'

            deadline = monotonic() + 8
            while True:
                after = (await go.get(view_path)).raise_for_status().json()
                if after != before:
                    break
                assert monotonic() < deadline, after
                await asyncio.sleep(.05)
            assert {key: val for key, val in after.items() if key != 'predictions'} == {
                key: val for key, val in before.items() if key != 'predictions'
            }
            expected = next(point for point in before['predictions']
                            if datetime.fromisoformat(point['valid_time']) == target)
            assert expected['variables']['kp']['binary'][1]['probability'] == .05
            assert expected['variables']['ap']['continuous']['q50'] == 6
            replacement = next(point for point in after['predictions']
                               if datetime.fromisoformat(point['valid_time']) == target)
            assert datetime.fromisoformat(replacement['update_issued_at']) == fast_issue
            assert replacement['lead_hours'] == expected['lead_hours']
            assert replacement['variables']['kp']['binary'][1]['probability'] == .7
            assert replacement['variables']['ap']['continuous'] == {'q10': 9, 'q50': 32, 'q90': 80}
            assert [point for point in after['predictions'] if point is not replacement] == [
                point for point in before['predictions'] if point is not expected
            ]
            assert ForecastRelease.model_validate((await go.get(archive_path)).raise_for_status().json()) == archived


async def test_fast_bt_updates_hmf_without_replacing_bs(recorder_setup, monkeypatch):
    import asyncio

    from common.schemas.forecast_release import FAST_BT_ARTIFACT, FAST_BT_COLUMNS, FAST_BT_PRODUCT, PREDICTION_COLUMNS

    _, _, config, writer = recorder_setup
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-token')
    monkeypatch.setattr(common_config, '_config', None)
    headers = {'Authorization': 'Bearer test-token'}
    base_issue = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    base = await RunRecorder.begin('hmf', 'manual', config, writer=writer)
    for name, values in [('hmf_total_threshold', '.3,.2,.1'), ('hmf_southward_threshold', '.2,.1,.05')]:
        columns = ['issue_time', 'valid_time', 'lead_hours', *PREDICTION_COLUMNS[name]]
        rows = [f'{base_issue.isoformat()},{(base_issue + timedelta(hours=lead)).isoformat()},{lead},{values}'
                for lead in range(1, 13)]
        await base.store(name, ('\n'.join([','.join(columns), *rows]) + '\n').encode(), {}, len(rows), columns)
    release = await base.finish()
    view_path = '/internal/v1/public-forecasts/hmf'
    archive_path = f'/internal/v1/forecasts/hmf/releases/{release.release_id}'
    report_path = 'configs/evaluations/hmf-fast-l1.json'
    (config.workdir / report_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / report_path, config.workdir / report_path)
    shutil.copyfile(ROOT / 'configs/models_registry.yaml', config.workdir / 'configs/models_registry.yaml')

    with go_server(config.workdir) as (_, address):
        async with httpx.AsyncClient(base_url=address, headers=headers, trust_env=False) as go:
            metrics = (await go.get('/internal/v1/public-forecast-metrics/hmf')).raise_for_status().json()
            assert metrics['fast_bt_evaluation'] == json.loads((ROOT / report_path).read_text())
            deadline = monotonic() + 8
            while True:
                response = await go.get(view_path)
                if response.status_code != 503 or monotonic() >= deadline:
                    before = response.raise_for_status().json()
                    break
                await asyncio.sleep(.05)

            issue = datetime.now(UTC)
            hour = issue.replace(minute=0, second=0, microsecond=0)
            targets = {hour + timedelta(hours=2), hour + timedelta(hours=3)}
            columns = ['issue_time', 'valid_time', *FAST_BT_COLUMNS]
            rows = [f'{issue.isoformat()},{target.isoformat()},.8,.5,.3' for target in sorted(targets)]
            content = '\n'.join([','.join(columns), *rows, ''])
            fast = await RunRecorder.begin(FAST_BT_PRODUCT, 'scheduled', config, writer=writer)
            await fast.store(FAST_BT_ARTIFACT, content.encode(), {}, len(rows), columns)
            assert (await fast.finish()).status == 'published'

            deadline = monotonic() + 8
            while True:
                after = (await go.get(view_path)).raise_for_status().json()
                if after != before:
                    break
                assert monotonic() < deadline, after
                await asyncio.sleep(.05)
            assert after['issue_time'] == before['issue_time']
            assert after['horizon_hours'] == before['horizon_hours']
            for current, expected in zip(after['predictions'], before['predictions'], strict=True):
                target = datetime.fromisoformat(expected['valid_time'])
                if target not in targets:
                    assert current == expected
                    continue
                assert current['lead_hours'] == expected['lead_hours']
                assert current['variables']['bs'] == expected['variables']['bs']
                assert 'update_issued_at' not in current
                bt = current['variables']['bt']
                assert datetime.fromisoformat(bt['update_issued_at']) == issue
                assert [item['probability'] for item in bt['binary']] == [.8, .5, .3]

            archived = ForecastRelease.model_validate((await go.get(archive_path)).raise_for_status().json())
            assert archived == await read_release('hmf', release.release_id)
            assert archived.issue_time == base_issue


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

                async def read_forecast(target, accept=lambda _: True):
                    deadline = monotonic() + 8
                    while True:
                        response = await public.get(f'/api/v1/forecasts/{target}')
                        if response.status_code == 200:
                            data = response.json()['data']
                            if accept(data):
                                return data
                        assert monotonic() < deadline, response.text
                        await asyncio.sleep(0.05)

                for target in targets:
                    assert await read_forecast(target) == reference[target]
                assert await read_forecast('solar-wind-events') == event_report
                metrics = await public.get('/api/v1/forecasts/dst/metrics')
                assert metrics.status_code == 200
                assert metrics.json()['data'] == reference['dst-metrics']
                assert (await public.get('/api/v1/forecasts/hmf/metrics')).status_code == 503

                updated_issue = issue + timedelta(hours=1)
                updated = await RunRecorder.begin('dst', 'manual', config, writer=writer)
                await store_product(updated, issue=updated_issue.isoformat())
                await updated.finish()
                received = await read_forecast('dst', lambda data: datetime.fromisoformat(data['issue_time']) == updated_issue)
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
    'dst': products.load_forecast(products.PRODUCTS['dst']).model_dump(mode='json'),
    'atmospheric-density': DensityForecast.from_internal(products.load_density_forecast()).model_dump(mode='json'),
    'dst-metrics': products.load_metrics(products.PRODUCTS['dst']).model_dump(mode='json'),
}))
"""
    result = subprocess.run([ROOT / 'apps/api/.venv/bin/python', '-c', code, str(workdir)], cwd=ROOT,
                            input=json.dumps(releases), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
