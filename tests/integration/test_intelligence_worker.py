"""Real ownership, deduplication, retry and crash recovery on disposable PostgreSQL."""
from uuid import uuid4

import pytest
psycopg = pytest.importorskip('psycopg')
from sqlalchemy.engine import make_url

from test_domain_storage import database, migrate
from argus_intelligence import worker


@pytest.fixture
def intelligence_db(database, monkeypatch):
    dsn, passwords, environment = database
    migrate(environment)
    for key, value in environment.items():
        if key.startswith('INTELLIGENCE_DB_'):
            monkeypatch.setenv(key, value)
    return dsn['intelligence'], passwords


def evidence(release_id=None):
    return {'release_id': str(release_id or uuid4()), 'run_id': str(uuid4()), 'mode': 'stub', 'risk_assessment': None}


def test_completion_survives_restart_and_result_is_atomic(intelligence_db):
    item = evidence()
    fetch = lambda *args, **kwargs: item
    assert worker.process_once(fetch=fetch)['status'] == 'succeeded'
    assert worker.process_once(fetch=fetch, assess=lambda _: pytest.fail('Duplicate computation'))['status'] == 'skipped'
    status = worker.status()
    assert status['latest_result']['result'] == item
    assert status['latest_attempt']['status'] == 'skipped'
    assert status['pending_releases'] == 0


def test_failed_processing_retries_pinned_release_before_newer_latest(intelligence_db):
    item = evidence()
    def fail(_): raise ValueError('sensitive input')
    with pytest.raises(ValueError):
        worker.process_once(fetch=lambda *a, **kw: item, assess=fail)
    assert worker.status()['latest_result'] is None
    def retry(product, release_id=None):
        assert str(release_id) == item['release_id']
        return item
    assert worker.process_once(fetch=retry)['status'] == 'succeeded'
    assert 'sensitive' not in str(worker.status())


def test_interrupted_attempt_recovers_and_lock_excludes_other_writer(intelligence_db):
    item = evidence()
    with worker.db.connect() as conn:
        conn.execute('SELECT pg_advisory_lock(%s,%s)', worker.LOCK)
        assert worker.process_once(fetch=lambda *a, **kw: pytest.fail('Busy writer fetched data'))['status'] == 'busy'
        conn.execute("INSERT INTO intelligence.attempt(id,product,release_id,status) VALUES (%s,'solar-wind-speed',%s,'running')",
                     (uuid4(), item['release_id']))
    def fetch(product, release_id=None):
        assert str(release_id) == item['release_id']
        return item
    assert worker.process_once(fetch=fetch)['status'] == 'succeeded'
    with worker.db.connect() as conn:
        statuses = [row['status'] for row in conn.execute('SELECT status FROM intelligence.attempt').fetchall()]
        assert set(statuses) == {'interrupted', 'succeeded'}


def test_lost_session_cannot_commit_a_result(intelligence_db):
    dsn, _ = intelligence_db
    def terminate(item):
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE application_name='argus-intelligence' AND datname=current_database()")
        return item
    item = evidence()
    with pytest.raises(psycopg.Error):
        worker.process_once(fetch=lambda *a, **kw: item, assess=terminate)
    assert worker.status()['latest_result'] is None
    assert worker.process_once(fetch=lambda *a, **kw: item)['status'] == 'succeeded'


def test_domain_owner_has_ddl_and_foreign_databases_reject_connections(intelligence_db):
    dsn, urls = intelligence_db
    with worker.db.connect() as conn:
        conn.execute('CREATE TABLE intelligence.owner_test(id int)')
        conn.execute('DROP TABLE intelligence.owner_test')
    for domain in ('api', 'clio', 'prophet'):
        url = make_url(urls[domain]).set(database=make_url(dsn).database)
        with pytest.raises(psycopg.OperationalError):
            psycopg.connect(url.render_as_string(hide_password=False))


def test_failed_success_update_rolls_back_result_and_retries(intelligence_db):
    dsn, _ = intelligence_db
    with psycopg.connect(dsn) as admin:
        admin.execute("""CREATE FUNCTION intelligence.reject_success() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN IF NEW.status='succeeded' THEN RAISE EXCEPTION 'forced commit failure'; END IF; RETURN NEW; END $$""")
        admin.execute('CREATE TRIGGER reject_success BEFORE UPDATE ON intelligence.attempt FOR EACH ROW EXECUTE FUNCTION intelligence.reject_success()')
    item = evidence()
    with pytest.raises(psycopg.Error):
        worker.process_once(fetch=lambda *a, **kw: item)
    status = worker.status()
    assert status['latest_result'] is None
    assert status['latest_attempt']['status'] == 'failed'
    with psycopg.connect(dsn) as admin:
        admin.execute('DROP TRIGGER reject_success ON intelligence.attempt')
        admin.execute('DROP FUNCTION intelligence.reject_success()')
    assert worker.process_once(fetch=lambda *a, **kw: item)['status'] == 'succeeded'
