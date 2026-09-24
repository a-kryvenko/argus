"""Per-product scheduling, transaction boundaries and crash recovery on PostgreSQL."""
from test_prophet_ledger import recorder_database, database, store_product
from test_domain_storage import runtime
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import psycopg
import pytest

from argus_prophet.db.session import require_writer
from argus_prophet.services.runs import RunRecorder
from argus_prophet.services.generation.products import PRODUCTS
from argus_prophet.services.releases.publication import read_release, ReleaseNotFound
from argus_prophet.worker import generation_lock, run_due, due_slot, GenerationBusy
from argus_prophet.services.generation import cycle as generation_cycle

NOW = datetime(2026, 9, 13, 10, 10, tzinfo=UTC)


def generate(config, *, fail=(), seen=None):
    def calculate(slot, products):
        failures = {}
        for product in products:
            if seen is not None:
                seen.append(product)
            run = RunRecorder.begin(product, 'scheduled', config, scheduled_slot=slot)
            if product in fail:
                error = ValueError('model unavailable')
                run.finish(error=error)
                failures[product] = error
            else:
                store_product(run, names=PRODUCTS[product].artifacts, issue=slot.isoformat())
                run.finish()
        if failures:
            raise generation_cycle.GenerationFailed(failures)
    return calculate


def test_partial_cycle_publishes_successes_and_retries_only_failures(recorder_database):
    dsn, passwords, config = recorder_database
    with generation_lock():
        manual = RunRecorder.begin('dst', 'manual', config)
        store_product(manual)
        manual.finish()
    old_dst = read_release('dst')
    failed = {'dst', 'atmospheric-density'}
    with pytest.raises(generation_cycle.GenerationFailed):
        run_due(generate(config, fail=failed), NOW)
    assert read_release('dst') == old_dst
    speed = read_release('solar-wind-speed')
    seen = []
    assert run_due(generate(config, seen=seen), NOW)
    assert set(seen) == failed
    assert read_release('solar-wind-speed') == speed
    assert read_release('dst').run_id != manual.run_id
    assert not run_due(lambda *_: pytest.fail('Completed slot repeated'), NOW)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status,attempts,error FROM prophet.forecast_slot').fetchone() == ('succeeded', 8, None)
        assert conn.execute('SELECT scheduled_slot FROM prophet.forecast_run WHERE id=%s', (manual.run_id,)).fetchone()[0] is None


def test_new_hour_drops_old_retries_and_does_not_replay_missed_hours(recorder_database):
    dsn, passwords, config = recorder_database
    with pytest.raises(generation_cycle.GenerationFailed):
        run_due(generate(config, fail=('dst',)), NOW)
    seen = []
    assert run_due(generate(config, seen=seen), NOW + timedelta(hours=3))
    assert seen == list(PRODUCTS)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT slot,status FROM prophet.forecast_slot ORDER BY slot').fetchall() == [
            (due_slot(NOW), 'partial'), (due_slot(NOW) + timedelta(hours=3), 'succeeded')]


def test_all_failed_products_retry_and_keep_history(recorder_database):
    dsn, passwords, config = recorder_database
    with pytest.raises(generation_cycle.GenerationFailed):
        run_due(generate(config, fail=PRODUCTS), NOW)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_slot').fetchone()[0] == 'failed'
    assert run_due(generate(config), NOW)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status,attempts FROM prophet.forecast_slot').fetchone() == ('succeeded', 12)
        assert conn.execute("SELECT count(*) FROM prophet.forecast_run WHERE status='failed'").fetchone()[0] == 6


def test_crash_between_products_keeps_committed_product_and_recovers_next(recorder_database, tmp_path):
    _, _, config = recorder_database
    first, second, *_ = PRODUCTS
    def crash(slot, products):
        generate(config)(slot, (first,))
        RunRecorder.begin(second, 'scheduled', config, scheduled_slot=slot)
        raise RuntimeError('crashed before second product finished')
    with pytest.raises(RuntimeError):
        run_due(crash, NOW)
    release = read_release(first)
    seen = []
    other = SimpleNamespace(workdir=tmp_path / 'new-workdir', models_registry={})
    assert run_due(generate(other, seen=seen), NOW)
    assert set(seen) == set(PRODUCTS) - {first}
    assert read_release(first) == release


def test_crash_after_final_commit_does_not_repeat_slot(recorder_database):
    _, _, config = recorder_database
    def crash(slot, products):
        generate(config)(slot, products)
        raise RuntimeError('crash after commit')
    with pytest.raises(RuntimeError):
        run_due(crash, NOW)
    assert not run_due(lambda *_: pytest.fail('Committed slot repeated'), NOW)


def test_independent_sessions_cannot_generate_concurrently(recorder_database):
    with generation_lock(), ThreadPoolExecutor(max_workers=1) as pool:
        def contender():
            with generation_lock():
                pytest.fail('Competing writer acquired the lock')
        with pytest.raises(GenerationBusy):
            pool.submit(contender).result(timeout=10)
    with generation_lock():
        assert require_writer().execute('SELECT 1').fetchone()[0] == 1


def test_disconnected_writer_cannot_publish_after_new_owner(recorder_database):
    dsn, passwords, config = recorder_database
    with generation_lock():
        old = RunRecorder.begin('dst', 'scheduled', config, scheduled_slot=due_slot(NOW))
        store_product(old)
        pid = require_writer().info.backend_pid
        with psycopg.connect(dsn['prophet'], autocommit=True) as admin:
            assert admin.execute('SELECT pg_terminate_backend(%s)', (pid,)).fetchone()[0]
        with pytest.raises((psycopg.Error, RuntimeError)):
            old.finish()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(run_due, generate(config), NOW).result(timeout=30)
        with pytest.raises((psycopg.Error, RuntimeError)):
            old.finish()
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (old.run_id,)).fetchone()[0] == 'interrupted'
        assert conn.execute('SELECT count(*) FROM prophet.forecast_release').fetchone()[0] == 6
    assert read_release('dst').run_id != old.run_id


def test_publication_and_product_completion_roll_back_together(recorder_database):
    dsn, passwords, config = recorder_database
    with generation_lock():
        run = RunRecorder.begin('dst', 'scheduled', config, scheduled_slot=due_slot(NOW))
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
    assert run_due(generate(config), NOW)


def test_legacy_partial_batch_retries_only_unpublished_products(recorder_database):
    dsn, passwords, config = recorder_database
    # Reproduce a historical all-products run without changing historical migrations.
    with generation_lock():
        run = RunRecorder.begin('dst', 'scheduled', config, scheduled_slot=due_slot(NOW))
        store_product(run)
        run.finish()
    with runtime(dsn, 'prophet', passwords) as conn:
        conn.execute("UPDATE prophet.forecast_run SET product='all',status='partial' WHERE id=%s", (run.run_id,))
    seen = []
    assert run_due(generate(config, seen=seen), NOW)
    assert set(seen) == set(PRODUCTS) - {'dst'}
    assert read_release('dst').run_id == run.run_id


def test_real_cycle_records_shared_inputs_and_retries_failed_product(recorder_database, monkeypatch):
    from common import config as common_config
    from common.schemas.forecast_inputs import ForecastInputs
    from common.schemas.observation import Observation
    from argus_prophet.services import inputs as observations
    from argus_prophet.services.generation import calculation as generation
    from unittest.mock import Mock
    dsn, passwords, config = recorder_database
    monkeypatch.setattr(common_config, 'get_config', lambda: config)
    inputs = ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[]))
    load = Mock(return_value=inputs)
    monkeypatch.setattr(observations, 'load_inputs', load)
    failed = {'dst'}
    seen = []
    def calculate(product, *, inputs, recorder):
        seen.append(product)
        if product in failed:
            raise ValueError('model unavailable')
        store_product(recorder, names=PRODUCTS[product].artifacts, issue=due_slot(NOW).isoformat())
    monkeypatch.setattr(generation, 'calculate', calculate)
    def cycle(slot, products):
        generation_cycle.generate_products(products, 'scheduled', scheduled_slot=slot)
    with pytest.raises(generation_cycle.GenerationFailed):
        run_due(cycle, NOW)
    assert seen == list(PRODUCTS)
    assert load.call_count == 1
    speed = read_release('solar-wind-speed')
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT count(DISTINCT input_sha256) FROM prophet.forecast_run').fetchone()[0] == 1
        assert conn.execute("SELECT status FROM prophet.forecast_run WHERE product='dst'").fetchone()[0] == 'failed'
    seen.clear()
    failed.clear()
    assert run_due(cycle, NOW)
    assert seen == ['dst'] and load.call_count == 2
    assert read_release('solar-wind-speed') == speed
    assert read_release('dst').product == 'dst'
