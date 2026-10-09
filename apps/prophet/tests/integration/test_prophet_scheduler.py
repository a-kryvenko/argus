"""Production dispatcher, transaction boundaries and crash recovery on PostgreSQL."""
from .storage import recorder_database, database, store_product
from domain_storage import runtime
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import psycopg
import pytest

from argus_prophet.services.runs import RunRecorder
from argus_prophet.services.generation.products import PRODUCTS
from argus_prophet.services.releases.publication import read_release, ReleaseNotFound
from argus_prophet.scheduling.jobs import generation_lock, GenerationBusy
from argus_prophet.config import ProductSchedule, ProphetConfig
from argus_prophet.scheduling.execution import ExecutionControl
from argus_prophet.services.generation.calculation import Artifact
from argus_prophet import worker

NOW = datetime(2026, 9, 13, 10, 10, tzinfo=UTC)
due_slot = ProductSchedule().due_slot


def dispatch(config, now=NOW, *, fail=(), seen=None):
    """Real scheduler and DB; replace only process/model work with ready futures."""
    from common.schemas.forecast_inputs import ForecastInputs
    from common.schemas.observation import Observation
    class Pool:
        def submit(self, execute, target, *args, **kwargs):
            future = Future()
            try:
                if getattr(target, 'func', target) is worker.load_inputs:
                    result = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
                else:
                    product = args[0].product
                    if seen is not None:
                        seen.append(product)
                    if product in fail:
                        raise ValueError('model unavailable')
                    result = []
                    recorder = SimpleNamespace(store=lambda *values: result.append(Artifact(*values)))
                    store_product(recorder, names=PRODUCTS[product].artifacts, issue=due_slot(now).isoformat())
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
            return future
        def shutdown(self, **kwargs):
            pass
    config.models_registry.setdefault('models', {})
    d = worker.Dispatcher(ProphetConfig(), config, ExecutionControl(), Pool())
    try:
        # Each turn drains the two phases and gives the next products capacity.
        for _ in range(len(PRODUCTS)):
            d.launch_due(now, 0)
            d.finish_tasks()
            d.finish_tasks()
    finally:
        d.close()


def test_partial_cycle_publishes_successes_and_retries_only_failures(recorder_database):
    dsn, passwords, config = recorder_database
    with generation_lock() as writer:
        manual = RunRecorder.begin('dst', 'manual', config, writer=writer)
        store_product(manual)
        manual.finish()
    old_dst = read_release('dst')
    failed = {'dst', 'atmospheric-density'}
    dispatch(config, fail=failed)
    assert read_release('dst') == old_dst
    speed = read_release('solar-wind-speed')
    seen = []
    dispatch(config, seen=seen)
    assert set(seen) == failed
    assert read_release('solar-wind-speed') == speed
    assert read_release('dst').run_id != manual.run_id
    seen.clear()
    dispatch(config, seen=seen)
    assert not seen
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT product,status,attempts,error FROM prophet.forecast_slot ORDER BY product').fetchall() == [
            (p, 'succeeded', 2 if p in failed else 1, None) for p in sorted(PRODUCTS)]
        assert conn.execute('SELECT scheduled_slot FROM prophet.forecast_run WHERE id=%s', (manual.run_id,)).fetchone()[0] is None
        assert conn.execute('SELECT count(DISTINCT input_sha256) FROM prophet.forecast_run WHERE scheduled_slot IS NOT NULL').fetchone()[0] == 1


def test_new_hour_drops_old_retries_and_does_not_replay_missed_hours(recorder_database):
    dsn, passwords, config = recorder_database
    dispatch(config, fail=('dst',))
    seen = []
    dispatch(config, NOW + timedelta(hours=3), seen=seen)
    assert seen == list(PRODUCTS)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT product,slot,status FROM prophet.forecast_slot ORDER BY slot,product').fetchall() == [
            (p, due_slot(NOW), 'failed' if p == 'dst' else 'succeeded') for p in sorted(PRODUCTS)] + [
            (p, due_slot(NOW) + timedelta(hours=3), 'succeeded') for p in sorted(PRODUCTS)]


def test_all_failed_products_retry_and_keep_history(recorder_database):
    dsn, passwords, config = recorder_database
    dispatch(config, fail=PRODUCTS)
    dispatch(config)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status,attempts FROM prophet.forecast_slot').fetchall() == [('succeeded', 2)] * 6
        assert conn.execute("SELECT count(*) FROM prophet.forecast_run WHERE status='failed'").fetchone()[0] == 6


def test_crash_between_products_keeps_committed_product_and_recovers_next(recorder_database, tmp_path):
    _, _, config = recorder_database
    first, second, *_ = PRODUCTS
    with generation_lock() as writer:
        run = RunRecorder.begin(first, 'scheduled', config, writer=writer, scheduled_slot=due_slot(NOW))
        store_product(run, names=PRODUCTS[first].artifacts)
        run.finish()
        unfinished = RunRecorder.begin(second, 'scheduled', config, writer=writer, scheduled_slot=due_slot(NOW))
    release = read_release(first)
    seen = []
    other = SimpleNamespace(workdir=tmp_path / 'new-workdir', models_registry={})
    dispatch(other, seen=seen)
    assert set(seen) == set(PRODUCTS) - {first}
    assert read_release(first) == release
    with generation_lock() as writer:
        assert writer.execute('SELECT status FROM prophet.forecast_run WHERE id=%s',
                                        (unfinished.run_id,)).fetchone()[0] == 'interrupted'


def test_independent_sessions_cannot_generate_concurrently(recorder_database):
    with generation_lock() as writer, ThreadPoolExecutor(max_workers=1) as pool:
        def contender():
            with generation_lock() as writer:
                pytest.fail('Competing writer acquired the lock')
        with pytest.raises(GenerationBusy):
            pool.submit(contender).result(timeout=10)
    with generation_lock() as writer:
        assert writer.execute('SELECT 1').fetchone()[0] == 1


def test_disconnected_writer_cannot_publish_after_new_owner(recorder_database):
    dsn, passwords, config = recorder_database
    with generation_lock() as writer:
        old = RunRecorder.begin('dst', 'scheduled', config, writer=writer, scheduled_slot=due_slot(NOW))
        store_product(old)
        pid = writer.info.backend_pid
        with psycopg.connect(dsn['prophet'], autocommit=True) as admin:
            assert admin.execute('SELECT pg_terminate_backend(%s)', (pid,)).fetchone()[0]
        with pytest.raises((psycopg.Error, RuntimeError)):
            old.finish()
        # Even a new writer in the same thread cannot rebind the old recorder.
        dispatch(config)
        with pytest.raises((psycopg.Error, RuntimeError)):
            old.finish()
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (old.run_id,)).fetchone()[0] == 'interrupted'
        assert conn.execute('SELECT count(*) FROM prophet.forecast_release').fetchone()[0] == 6
    assert read_release('dst').run_id != old.run_id


def test_publication_and_product_completion_roll_back_together(recorder_database):
    dsn, passwords, config = recorder_database
    with generation_lock() as writer:
        run = RunRecorder.begin('dst', 'scheduled', config, writer=writer, scheduled_slot=due_slot(NOW))
        store_product(run)
        with runtime(dsn, 'prophet', passwords) as conn:
            conn.execute('UPDATE prophet.forecast_artifact SET sha256=%s', ('0' * 64,))
        with pytest.raises(ValueError, match='checksum'):
            run.finish()
        with pytest.raises(ReleaseNotFound):
            read_release('dst')
        with runtime(dsn, 'prophet', passwords) as conn:
            assert conn.execute('SELECT status FROM prophet.forecast_slot').fetchone()[0] == 'running'
            assert conn.execute('SELECT status FROM prophet.forecast_run').fetchone()[0] == 'running'
    dispatch(config)
    assert read_release('dst').run_id != run.run_id


def test_legacy_partial_batch_retries_only_unpublished_products(recorder_database):
    dsn, passwords, config = recorder_database
    with generation_lock() as writer:
        run = RunRecorder.begin('dst', 'scheduled', config, writer=writer, scheduled_slot=due_slot(NOW))
        store_product(run)
        run.finish()
    with runtime(dsn, 'prophet', passwords) as conn:
        conn.execute("UPDATE prophet.forecast_run SET product='all',status='partial' WHERE id=%s", (run.run_id,))
    seen = []
    dispatch(config, seen=seen)
    assert set(seen) == set(PRODUCTS) - {'dst'}
    assert read_release('dst').run_id == run.run_id


def test_product_slots_allow_subhour_retries_and_independent_clock_rollback(recorder_database):
    from argus_prophet.scheduling.jobs import product_pending
    dsn, passwords, config = recorder_database
    slot = NOW.replace(minute=5)
    with generation_lock() as writer:
        assert product_pending('dst', slot, writer=writer)
        run = RunRecorder.begin('dst', 'scheduled', config, writer=writer, scheduled_slot=slot)
        store_product(run)
        run.finish()
        assert not product_pending('dst', slot, writer=writer)
        assert not product_pending('dst', slot-timedelta(minutes=5), writer=writer)
        assert product_pending('hmf', slot, writer=writer)
        assert product_pending('dst', slot+timedelta(minutes=5), writer=writer)
        failed = RunRecorder.begin('dst', 'scheduled', config, writer=writer, scheduled_slot=slot+timedelta(minutes=5))
        failed.finish(error=ValueError('model timeout'))
        assert product_pending('dst', slot+timedelta(minutes=5), writer=writer)
    with generation_lock() as writer:
        assert not product_pending('dst', slot, writer=writer)
        assert product_pending('dst', slot+timedelta(minutes=5), writer=writer)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT product,slot,status FROM prophet.forecast_slot ORDER BY slot').fetchall() == [
            ('dst', slot, 'succeeded'), ('dst', slot+timedelta(minutes=5), 'failed')]


def test_verification_excludes_generation_and_cleanup(recorder_database):
    from argus_prophet.scheduling.jobs import verification_lock, retention_lock
    import threading
    acquired, release = threading.Event(), threading.Event()
    def verify():
        with verification_lock() as writer:
            writer.execute("INSERT INTO prophet.scheduled_job VALUES ('test', now(), now())")
            acquired.set()
            assert release.wait(10)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(verify)
        try:
            assert acquired.wait(10)
            with pytest.raises(GenerationBusy):
                with generation_lock():
                    pytest.fail('Generation overlapped verification')
            with pytest.raises(GenerationBusy):
                with retention_lock() as writer:
                    pytest.fail('Cleanup overlapped verification')
        finally:
            release.set()
            future.result(timeout=10)
    with retention_lock() as writer:
        writer.execute('SELECT 1')


def test_generation_excludes_verification_and_releases_locks_after_failure(recorder_database):
    from argus_prophet.scheduling.jobs import verification_lock, retention_lock
    with pytest.raises(ValueError, match='calculation failed'):
        with generation_lock():
            with pytest.raises(GenerationBusy):
                with verification_lock():
                    pytest.fail('Verification overlapped generation')
            raise ValueError('calculation failed')
    with pytest.raises(ValueError, match='verification failed'):
        with verification_lock() as writer:
            writer.execute('SELECT 1')
            raise ValueError('verification failed')
    # Neither the failed acquisition nor either task failure leaks a lock.
    with generation_lock() as writer:
        writer.execute('SELECT 1')
    with verification_lock() as writer:
        writer.execute('SELECT 1')
    with retention_lock() as writer:
        writer.execute('SELECT 1')


def test_product_slot_migration_preserves_aggregate_history(database, monkeypatch):
    import subprocess
    import sys
    from uuid import uuid4
    from domain_storage import ROOT
    from argus_prophet.scheduling.jobs import product_pending
    dsn, passwords, environment = database
    command = [sys.executable, '-c', 'from argus_prophet.cli import main; main()', 'migrate', 'upgrade']
    result = subprocess.run([*command, '20260929_prophet_jobs'], env=environment, cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    run_id = uuid4()
    with runtime(dsn, 'prophet', passwords) as conn:
        conn.execute("INSERT INTO prophet.forecast_slot(slot,status,attempts) VALUES (%s,'partial',1)", (due_slot(NOW),))
        conn.execute("""INSERT INTO prophet.forecast_run
            (id,scope,product,trigger,started_at,status,provenance,scheduled_slot)
            VALUES (%s,'legacy','all','scheduled',%s,'partial','{}',%s)""", (run_id, NOW, due_slot(NOW)))
        conn.execute("""INSERT INTO prophet.forecast_release(id,product,run_id,published_at,issue_time,artifact_names)
            VALUES (%s,'dst',%s,%s,%s,'["dst_quantile"]')""", (uuid4(), run_id, NOW, due_slot(NOW)))
    result = subprocess.run([*command, 'head'], env=environment, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for key, value in environment.items():
        if key.startswith('PROPHET_DB_'):
            monkeypatch.setenv(key, value)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT product,slot,status,attempts FROM prophet.forecast_slot').fetchone() == (
            'all', due_slot(NOW), 'partial', 1)
        assert conn.execute('SELECT scheduled_product,scheduled_slot,status FROM prophet.forecast_run').fetchone() == (
            'all', due_slot(NOW), 'partial')
    with generation_lock() as writer:
        assert not product_pending('dst', due_slot(NOW), writer=writer)
        assert product_pending('hmf', due_slot(NOW), writer=writer)
        assert product_pending('dst', NOW.replace(minute=15), writer=writer)


def test_legacy_batch_is_readable_but_cannot_be_finished_or_republished(recorder_database):
    from argus_prophet.services.releases.publication import publish_run
    from argus_prophet.db.session import transaction
    dsn, passwords, config = recorder_database
    with generation_lock() as writer:
        with pytest.raises(ValueError, match='one supported'):
            RunRecorder.begin('all', 'manual', config, writer=writer)
        run = RunRecorder.begin('dst', 'manual', config, writer=writer)
        store_product(run)
        run.finish()
        release = read_release('dst')
        with runtime(dsn, 'prophet', passwords) as conn:
            conn.execute("UPDATE prophet.forecast_run SET product='all' WHERE id=%s", (run.run_id,))
        with transaction(writer) as conn:
            assert publish_run(conn, run.run_id) is None
        assert read_release('dst') == release
        with runtime(dsn, 'prophet', passwords) as conn:
            conn.execute("UPDATE prophet.forecast_run SET status='running' WHERE id=%s", (run.run_id,))
        with pytest.raises(RuntimeError):
            run.finish()
        with runtime(dsn, 'prophet', passwords) as conn:
            assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s',
                                (run.run_id,)).fetchone()[0] == 'running'
        assert read_release('dst') == release
