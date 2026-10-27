from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import asyncio
"""Uses the explicitly configured disposable PostgreSQL server, never production."""
from tests.support.database import ROOT, database, runtime
from .storage import recorder_database, recorder_setup, store_product
import gzip
import hashlib
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from forecast.inputs import ForecastInputs
from common.schemas.observation import Observation
from prophet.services.releases.recording import RunRecorder
from prophet.services.releases.publication import read_release, ReleaseNotFound


@pytest.mark.parametrize(('speed_available', 'speed_integrated'), [(True, None), (True, 650), (False, None)])
async def test_cme_arrival_and_speed_snapshot_release_roundtrip(recorder_setup, speed_available, speed_integrated):
    import csv
    import io
    import json
    from uuid import uuid4
    from common.schemas.solar_wind_events import EventInputs, SourcePoll, SolarWindEvent, SolarWindEventForecast
    from prophet.services.events.calculation import calculate
    dsn, passwords, config, writer = recorder_setup
    now = datetime(2026, 10, 13, 12, tzinfo=UTC)
    rows = [dict(activityID='cme1', startTime='2026-10-13T00:00Z', cmeAnalyses=[dict(
        isMostAccurate=True, featureCode='LE', time21_5='2026-10-13T03:00Z', speed=900,
        latitude=0, longitude=5, halfAngle=45, submissionTime='2026-10-13T04:00Z')])]
    poll = SourcePoll(endpoint='CME', poll_id=uuid4(), received_at=now,
        sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(), request_start=now-timedelta(days=30),
        request_end=now, records=rows)
    arrival = now + timedelta(hours=24)
    simulations = [dict(cmeInputs=[dict(CMEID='cme1')], link='enlil-run1',
        modelCompletionTime=(now-timedelta(hours=1)).isoformat(),
        estimatedShockArrivalTime=arrival.isoformat())]
    enlil_poll = SourcePoll(endpoint='WSAEnlilSimulations', poll_id=uuid4(), received_at=now,
        sha256=hashlib.sha256(json.dumps(simulations).encode()).hexdigest(),
        request_start=now-timedelta(days=30), request_end=now, records=simulations)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]),
        solar_wind_events=EventInputs(as_of=now, read_at=now, polls=[poll, enlil_poll], events=[SolarWindEvent(event_id='cme1', kind='cme', status='predicted_hit', source='DONKI/WSA-ENLIL', source_ids=['cme1'], available_at=now, arrival_at=arrival, arrival_target='Earth', model_completed_at=now-timedelta(hours=1), initial_speed_km_s=900, simulation_id='enlil-run1')]))
    run = (await RunRecorder.begin('solar-wind-events', 'manual', config, writer=writer))
    (await run.snapshot(inputs))
    speed = dict(model_version='cme-speed-v2', state='available', model_sha256='a'*64,
        points=[dict(valid_time=arrival, speed_background=400, speed_nominal=800,
            speed_early=600, speed_late=400, scenario_lower=400, scenario_upper=1000)])
    if speed_integrated is not None:
        speed['points'][0]['speed_integrated'] = speed_integrated

    def speed_calculator(events):
        assert len(events) == 1 and events[0].arrival_at == arrival
        return speed

    artifact = calculate(inputs, speed_calculator=speed_calculator if speed_available else None)[0]
    (await run.store(artifact.name, artifact.content, artifact.model_info, artifact.row_count, artifact.columns))
    (await run.finish())
    release = (await read_release('solar-wind-events'))
    assert release.run_id == run.run_id
    assert release.artifacts[0].csv_text.encode() == artifact.content
    row = next(csv.DictReader(io.StringIO(release.artifacts[0].csv_text)))
    report = SolarWindEventForecast.model_validate_json(row['report_json'])
    assert report.model_version == 'solar-wind-events-v4'
    assert report.cme_ensembles == [] and report.hss_forecasts == []
    assert report.source_status['DBEM']['state'] == 'disabled'
    assert report.source_status['CME']['poll_id'] == str(poll.poll_id)
    assert report.source_status['WSAEnlilSimulations']['sha256'] == enlil_poll.sha256
    assert report.events[0].event_id == 'cme1'
    assert report.events[0].status == 'predicted_hit'
    assert report.events[0].arrival_at == arrival
    assert report.cme_speed.state == ('available' if speed_available else 'unavailable')
    if speed_available:
        assert report.cme_speed.model_dump(mode='json')['points'] == [
            {**speed['points'][0], 'valid_time': arrival.isoformat().replace('+00:00', 'Z'),
             'speed_integrated': speed_integrated}]
        assert report.cme_speed.model_sha256 == speed['model_sha256']
        assert report.cme_speed.scenario_kind == 'conditional_scenarios_not_quantiles'
    else:
        assert report.cme_speed.points == []
    with runtime(dsn, 'prophet', passwords) as conn:
        snapshot = conn.execute('SELECT input_snapshot FROM prophet.forecast_run WHERE id=%s',
                                (run.run_id,)).fetchone()[0]
        assert snapshot['solar_wind_events'] == inputs.model_dump(mode='json')['solar_wind_events']


async def test_snapshot_bytes_publication_and_role_boundary(recorder_setup):
    dsn, passwords, config, writer = recorder_setup
    run = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    now = datetime.now(UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
    (await run.snapshot(inputs))
    with pytest.raises(RuntimeError, match='once'):
        (await run.snapshot(inputs))
    content = (await store_product(run)).encode('utf-8')
    (await run.finish())
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (run.run_id,)).fetchone()[0] == 'succeeded'
        blob, digest = conn.execute('SELECT csv_gzip,sha256 FROM prophet.forecast_artifact WHERE run_id=%s', (run.run_id,)).fetchone()
        assert gzip.decompress(blob) == content and digest == hashlib.sha256(content).hexdigest()
        assert conn.execute("SELECT to_regclass('prophet.forecast_export')").fetchone()[0] is None
        assert not conn.execute("SELECT 1 FROM information_schema.columns WHERE table_schema='prophet' AND table_name='forecast_artifact' AND column_name='csv_written_at'").fetchone()
        for table in ('clio.measurement', 'api.dashboard_user'):
            with pytest.raises(psycopg.errors.UndefinedTable):
                conn.execute(f'SELECT * FROM {table}')
    for domain in ('api', 'clio'):
        with runtime(dsn, domain, passwords) as conn:
            with pytest.raises(psycopg.errors.UndefinedTable):
                conn.execute('SELECT * FROM prophet.forecast_run')
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT count(*) FROM prophet.forecast_release WHERE run_id=%s',
                            (run.run_id,)).fetchone()[0] == 1
    assert (await read_release('dst')).artifacts[0].csv_text.encode('utf-8') == content


async def test_interrupted_and_failed_runs_are_preserved(recorder_setup):
    dsn, passwords, config, writer = recorder_setup
    first = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    from prophet.db.locks import recover_interrupted
    (await recover_interrupted(writer))
    second = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await second.finish(error=ValueError('bad model')))
    with runtime(dsn, 'prophet', passwords) as conn:
        rows = dict(conn.execute('SELECT id,status FROM prophet.forecast_run').fetchall())
        assert rows[first.run_id] == 'interrupted'
        assert rows[second.run_id] == 'failed'


async def test_shutdown_attempt_is_recorded_as_interrupted(recorder_setup):
    from prophet.execution import ShutdownRequested
    dsn, passwords, config, writer = recorder_setup
    run = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await run.finish(error=ShutdownRequested('deadline')))
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s',
                            (run.run_id,)).fetchone()[0] == 'interrupted'


async def test_cleanup_preserves_current_release_and_latest_attempt(recorder_setup):
    from prophet.services.releases.retention import cleanup
    from psycopg.types.json import Jsonb
    dsn, passwords, config, writer = recorder_setup
    old = datetime.now(UTC)-timedelta(days=100)
    first = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await store_product(first))
    (await first.finish())
    first_release = (await read_release('dst'))
    current = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await store_product(current))
    (await current.finish())
    latest = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await latest.finish(error=ValueError('test failure')))
    with runtime(dsn, 'prophet', passwords) as conn:
        conn.execute('INSERT INTO prophet.forecast_verification VALUES (%s,%s,%s,%s)',
                     (first_release.release_id, 'dst_quantile', old, Jsonb({})))
        conn.execute('UPDATE prophet.forecast_run SET started_at=%s,finished_at=%s WHERE id=%s',
                     (old, old, first.run_id))
        conn.execute('UPDATE prophet.forecast_run SET started_at=%s,finished_at=%s WHERE id=%s',
                     (old+timedelta(seconds=1), old, current.run_id))
        conn.execute('UPDATE prophet.forecast_run SET started_at=%s,finished_at=%s WHERE id=%s',
                     (old+timedelta(seconds=2), old, latest.run_id))
    preview = (await cleanup())
    assert preview['runs'] == preview['releases'] == preview['artifacts'] == preview['verifications'] == 1
    assert not preview['applied']
    assert (await read_release('dst', first_release.release_id))
    result = (await cleanup(apply=True, writer=writer))
    assert result['runs'] == 1 and result['applied']
    assert (await read_release('dst')).run_id == current.run_id
    with pytest.raises(ReleaseNotFound):
        (await read_release('dst', first_release.release_id))
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s',
                            (latest.run_id,)).fetchone()[0] == 'failed'


async def test_only_completed_products_publish_and_history_survives(recorder_setup):
    _, _, config, writer = recorder_setup
    first = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await store_product(first))
    with pytest.raises(ReleaseNotFound):
        (await read_release('dst'))
    (await first.finish())
    old = (await read_release('dst'))
    failed = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await store_product(failed))
    (await failed.finish(error=ValueError('model failed')))
    assert (await read_release('dst')).release_id == old.release_id
    second = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await store_product(second))
    (await second.finish())
    assert (await read_release('dst')).run_id == second.run_id
    assert (await read_release('dst', old.release_id)) == old
    incomplete = (await RunRecorder.begin('solar-wind-speed', 'manual', config, writer=writer))
    (await store_product(incomplete, names=('plasma_speed_quantile',)))
    with pytest.raises(ValueError, match='Incomplete'):
        (await incomplete.finish())
    (await incomplete.finish(error=ValueError('missing threshold artifact')))
    with pytest.raises(ReleaseNotFound):
        (await read_release('solar-wind-speed'))


async def test_latest_reads_and_publication_reject_delayed_run(recorder_setup):
    from prophet.services.releases.status import product_status
    dsn, passwords, config, writer = recorder_setup
    first = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await store_product(first)
    await first.finish()
    delayed = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await store_product(delayed)
    current = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await store_product(current)
    await current.finish()
    current_release = await read_release('dst')
    assert (await read_release('dst')) == current_release
    assert (await product_status('dst')).current_release.release_id == current_release.release_id
    await delayed.finish()
    assert (await read_release('dst')) == current_release
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT count(*) FROM prophet.forecast_release WHERE run_id=%s',
                            (delayed.run_id,)).fetchone()[0] == 0
    next_run = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await store_product(next_run)
    await next_run.finish()
    newest = await read_release('dst')
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute("SELECT to_regclass('prophet.current_forecast')").fetchone()[0] is None
    assert newest.run_id == next_run.run_id


async def test_cleanup_removes_old_release_and_preserves_latest_published_release(recorder_setup):
    from prophet.services.releases.retention import cleanup
    dsn, passwords, config, writer = recorder_setup
    first = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await store_product(first)
    await first.finish()
    first_release = await read_release('dst')
    current = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await store_product(current)
    await current.finish()
    latest = await RunRecorder.begin('dst', 'manual', config, writer=writer)
    await latest.finish(error=ValueError('test failure'))
    old = datetime.now(UTC) - timedelta(days=100)
    with runtime(dsn, 'prophet', passwords) as conn:
        for index, run in enumerate((first, current, latest)):
            conn.execute('UPDATE prophet.forecast_run SET started_at=%s,finished_at=%s WHERE id=%s',
                         (old + timedelta(seconds=index), old, run.run_id))
    assert (await cleanup())['runs'] == 1
    assert (await cleanup(apply=True, writer=writer))['runs'] == 1
    assert (await read_release('dst')).run_id == current.run_id
    with pytest.raises(ReleaseNotFound):
        await read_release('dst', first_release.release_id)


async def test_publication_failure_does_not_roll_back_another_product(recorder_setup):
    dsn, passwords, config, writer = recorder_setup
    success = (await RunRecorder.begin('solar-wind-density', 'manual', config, writer=writer))
    (await store_product(success, names=('plasma_density_quantile',)))
    (await success.finish())
    failed = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await store_product(failed))
    with runtime(dsn, 'prophet', passwords) as conn:
        conn.execute("UPDATE prophet.forecast_artifact SET sha256=%s WHERE name='dst_quantile'", ('0' * 64,))
    with pytest.raises(ValueError, match='checksum'):
        (await failed.finish())
    assert (await read_release('solar-wind-density')).run_id == success.run_id
    with pytest.raises(ReleaseNotFound):
        (await read_release('dst'))
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (failed.run_id,)).fetchone()[0] == 'running'


async def test_status_reports_product_failure_not_unrelated_success(recorder_setup):
    from prophet.services.releases.status import product_status
    _, _, config, writer = recorder_setup
    run = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    now = datetime(2026, 9, 13, 1, tzinfo=UTC)
    (await run.snapshot(ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))))
    (await store_product(run))
    (await run.finish())
    failure = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    (await failure.finish(error=ValueError('model unavailable')))
    unrelated = (await RunRecorder.begin('solar-wind-density', 'manual', config, writer=writer))
    (await store_product(unrelated, names=('plasma_density_quantile',)))
    (await unrelated.finish())
    status = (await product_status('dst', now=now))
    assert status.current_release.run_id == run.run_id
    assert status.current_release.input_diagnostics['normalized']['count'] == 0
    assert status.release_age_hours == 1 and status.freshness == 'unconfigured'
    assert status.latest_attempt.run_id == failure.run_id
    assert status.latest_attempt.status == 'failed'
    assert status.latest_attempt_artifacts == []
    assert (await product_status('solar-wind-density', now=now)).latest_attempt.run_id == unrelated.run_id


async def test_export_migration_preserves_release_bytes_on_upgrade_and_downgrade(recorder_setup):
    import importlib.util
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine
    from common.db.session import get_database_url
    _, _, config, writer = recorder_setup
    run = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
    expected = (await store_product(run))
    (await run.finish())
    release = (await read_release('dst'))
    path = ROOT / 'apps/prophet/src/prophet/db/migrations/versions/20260918_remove_exports.py'
    spec = importlib.util.spec_from_file_location('remove_exports', path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine(get_database_url('PROPHET'), connect_args={'options': '-csearch_path=prophet,pg_catalog,pg_temp'})
    try:
        with engine.begin() as conn:
            with Operations.context(MigrationContext.configure(conn)):
                migration.downgrade()
                assert conn.exec_driver_sql('SELECT count(*) FROM forecast_export').scalar() == 1
                migration.upgrade()
    finally:
        engine.dispose()
    assert (await read_release('dst')) == release
    assert (await read_release('dst')).artifacts[0].csv_text == expected


pytestmark = pytest.mark.anyio
