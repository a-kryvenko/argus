"""Real ownership, deduplication, retry and crash recovery on disposable PostgreSQL."""
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from common.db.session import dispose_engine
from intelligence.db import session as db
import pytest
psycopg = pytest.importorskip('psycopg')
from sqlalchemy.engine import make_url

from tests.support.database import database, migrate
from intelligence import worker


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


async def test_completion_survives_restart_and_result_is_atomic(intelligence_db):
    item = evidence()
    fetch = lambda *args, **kwargs: item
    assert (await worker.process_once(fetch=fetch))['status'] == 'succeeded'
    completed = await worker.status()
    attempt = completed['latest_attempt']
    assert attempt['finished_at'] >= attempt['started_at']
    assert completed['latest_result']['processed_at'] >= attempt['started_at']
    assert (await worker.process_once(fetch=fetch, assess=lambda _: pytest.fail('Duplicate computation')))['status'] == 'skipped'
    status = await worker.status()
    assert status['latest_result']['result'] == item
    assert status['latest_attempt']['status'] == 'skipped'
    assert status['latest_attempt']['finished_at'] >= status['latest_attempt']['started_at']
    assert status['pending_releases'] == 0


async def test_failed_processing_retries_pinned_release_before_newer_latest(intelligence_db):
    item = evidence()
    def fail(_): raise ValueError('sensitive input')
    with pytest.raises(ValueError):
        await worker.process_once(fetch=lambda *a, **kw: item, assess=fail)
    assert (await worker.status())['latest_result'] is None
    def retry(product, release_id=None):
        assert str(release_id) == item['release_id']
        return item
    assert (await worker.process_once(fetch=retry))['status'] == 'succeeded'
    assert 'sensitive' not in str((await worker.status()))


async def test_interrupted_attempt_recovers_and_lock_excludes_other_writer(intelligence_db):
    item = evidence()
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('SELECT pg_advisory_xact_lock(:p0,:p1)'), {'p0': worker.LOCK[0], 'p1': worker.LOCK[1]})
        assert (await worker.process_once(fetch=lambda *a, **kw: pytest.fail('Busy writer fetched data')))['status'] == 'busy'
        await session.execute(text("INSERT INTO intelligence.attempt(id,product,release_id,status) VALUES (:p0,'solar-wind-speed',:p1,'running')"), {'p0': uuid4(), 'p1': item['release_id']})
    def fetch(product, release_id=None):
        assert str(release_id) == item['release_id']
        return item
    assert (await worker.process_once(fetch=fetch))['status'] == 'succeeded'
    async with db.get_db_session() as session, session.begin():
        statuses = [row['status'] for row in (await session.execute(text('SELECT status FROM intelligence.attempt'))).mappings().all()]
        assert set(statuses) == {'interrupted', 'succeeded'}


async def test_lost_session_cannot_commit_a_result(intelligence_db):
    dsn, _ = intelligence_db
    def terminate(item):
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(pid) FROM pg_locks WHERE locktype='advisory' AND classid=736218 AND objid=3 AND granted AND database=(SELECT oid FROM pg_database WHERE datname=current_database())")
        return item
    item = evidence()
    with pytest.raises(SQLAlchemyError):
        await worker.process_once(fetch=lambda *a, **kw: item, assess=terminate)
    assert (await worker.status())['latest_result'] is None
    assert (await worker.process_once(fetch=lambda *a, **kw: item))['status'] == 'succeeded'


async def test_domain_owner_has_ddl_and_foreign_databases_reject_connections(intelligence_db):
    dsn, urls = intelligence_db
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('CREATE TABLE intelligence.owner_test(id int)'))
        await session.execute(text('DROP TABLE intelligence.owner_test'))
    for domain in ('api', 'clio', 'prophet'):
        url = make_url(urls[domain]).set(database=make_url(dsn).database)
        with pytest.raises(psycopg.OperationalError):
            psycopg.connect(url.render_as_string(hide_password=False))


async def test_failed_success_update_rolls_back_result_and_retries(intelligence_db):
    dsn, _ = intelligence_db
    with psycopg.connect(dsn) as admin:
        admin.execute("""CREATE FUNCTION intelligence.reject_success() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN IF NEW.status='succeeded' THEN RAISE EXCEPTION 'forced commit failure'; END IF; RETURN NEW; END $$""")
        admin.execute('CREATE TRIGGER reject_success BEFORE UPDATE ON intelligence.attempt FOR EACH ROW EXECUTE FUNCTION intelligence.reject_success()')
    item = evidence()
    with pytest.raises(SQLAlchemyError):
        await worker.process_once(fetch=lambda *a, **kw: item)
    status = await worker.status()
    assert status['latest_result'] is None
    assert status['latest_attempt']['status'] == 'failed'
    with psycopg.connect(dsn) as admin:
        admin.execute('DROP TRIGGER reject_success ON intelligence.attempt')
        admin.execute('DROP FUNCTION intelligence.reject_success()')
    assert (await worker.process_once(fetch=lambda *a, **kw: item))['status'] == 'succeeded'


pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
async def database_runtime(intelligence_db):
    try:
        yield
    finally:
        await dispose_engine()
