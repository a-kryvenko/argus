"""Real PostgreSQL ownership, revision races, immutable snapshots and event storage."""
from sqlalchemy import text

from datetime import UTC,datetime,timedelta
import hashlib
from unittest.mock import AsyncMock
import asyncio
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from common.db.session import dispose_engine, get_database_url
from intelligence.db import session as db
import pytest
from tests.support.database import database,migrate
from common.schemas.density_advisory import AdvisoryProfileInput,DensityAdvisory
from common.schemas.forecast_release import ForecastRelease
from intelligence import density_monitor as monitor
from intelligence.monitor import ProfileConflict,ProfileNotFound


@pytest.fixture
def storage(database,monkeypatch):
    _,_,environment=database
    migrate(environment)
    for key,value in environment.items():
        if key.startswith('INTELLIGENCE_DB_'): monkeypatch.setenv(key,value)
    monkeypatch.setattr(monitor,'max_age',lambda:6)
    monkeypatch.setattr(monitor,'prepare_assessment',canned)


def body():
    now=datetime.now(UTC)
    return AdvisoryProfileInput(name='Supplied satellite',trajectory=dict(source='Test ephemeris',generated_at=now,
        valid_until=now+timedelta(hours=1),coordinates='geocentric_spherical',points=[
            dict(time=now+timedelta(minutes=i),altitude_km=400,latitude_deg=0,longitude_deg=0) for i in (5,10)]))


def canned(profile,inputs,now):
    source=dict(trajectory_source=profile.trajectory.source,trajectory_generated_at=profile.trajectory.generated_at,
        cme_status='unavailable',density_release_id=getattr(inputs.get('atmospheric-density'),'release_id',None),
        density_issue_time=now,density_model='JB2008',reference=None)
    return DensityAdvisory(computed_at=now,attention='review',data_status='current',source=source,
        samples=[dict(**point.model_dump(),density_kg_m3=1e-12,ratio=3,attention='review',reasons=[]) for point in profile.trajectory.points],
        intervals=[],limitations=['Canned transport/storage evidence'])


def release():
    now = datetime.now(UTC)
    columns = ['issue_time', 'valid_time', 'lead_hours', 'altitude_km', 'latitude_deg',
               'rho_kg_m3', 'rho_lon_p10_kg_m3', 'rho_lon_p90_kg_m3']
    csv = ','.join(columns) + '\n' + f'{now.isoformat()},{now.isoformat()},0,400,0,1e-12,1e-12,1e-12\n'
    return ForecastRelease(release_id=uuid4(), run_id=uuid4(), product='atmospheric-density',
        published_at=now, issue_time=now, artifacts=[dict(name='atmospheric_density',
        sha256=hashlib.sha256(csv.encode()).hexdigest(), row_count=1, columns=columns,
        csv_text=csv, model_info={'name': 'Canned storage fixture'})])


async def test_all_spacecraft_are_always_monitored_and_migration_resumes_paused_profiles(storage, monkeypatch):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine
    from tests.support.database import ROOT, load

    paused = await monitor.save_profile(1, body().model_copy(update={'enabled': False}))
    active = await monitor.save_profile(1, body())
    assert paused.enabled is True
    async with db.get_db_session() as session, session.begin():
        await session.execute(text("UPDATE intelligence.density_profile SET payload=jsonb_set(payload,'{enabled}','false'::jsonb) WHERE id=:p0"), {'p0': paused.id})
    monkeypatch.setattr(monitor, 'retrieve_inputs', AsyncMock(return_value={}))
    assert (await monitor.refresh_profiles()) == {'profiles': 2, 'failed': 0}
    assert (await monitor.read_profile(1, paused.id)).enabled is True
    migration = load(ROOT / 'apps/intelligence/src/intelligence/db/migrations/versions/20261017_density_always_on.py')
    engine = create_engine(get_database_url('INTELLIGENCE'))
    try:
        with engine.begin() as connection:
            monkeypatch.setattr(migration, 'op', Operations(MigrationContext.configure(connection)))
            migration.upgrade()
            migration.upgrade()
    finally:
        engine.dispose()
    resumed = await monitor.read_profile(1, paused.id)
    assert resumed.enabled is True and resumed.revision == paused.revision + 1
    assert (await monitor.read_profile(1, active.id)).revision == active.revision
    async with db.get_db_session() as session, session.begin():
        row = (await session.execute(text('SELECT payload FROM intelligence.density_profile WHERE id=:p0'), {'p0': paused.id})).mappings().first()
        assert row['payload']['enabled'] is True
    assert (await monitor.refresh_profiles()) == {'profiles': 2, 'failed': 0}
    assert len((await monitor.history(1, paused.id))) == 2
    assert all(item.data_status != 'paused' for item in (await monitor.list_profiles(1)))


async def test_delete_ownership_revision_running_assessment_and_shared_inputs(storage):
    profile = await monitor.save_profile(1, body())
    other = await monitor.save_profile(2, body())
    source = release()
    inputs = {source.product: source}
    run = await monitor.assess_profile(1, profile.id, inputs=inputs)
    other_run = await monitor.assess_profile(2, other.id, inputs=inputs)
    with pytest.raises(ProfileNotFound):
        await monitor.delete_profile(2, profile.id, revision=profile.revision)
    with pytest.raises(ProfileConflict):
        await monitor.delete_profile(1, profile.id, revision=profile.revision+1)
    assert (await monitor.read_run(1, profile.id, run.id)).id == run.id
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('SELECT pg_advisory_xact_lock(hashtextextended(:p0,1))'), {'p0': str(profile.id)})
        with pytest.raises(ProfileConflict):
            await monitor.delete_profile(1, profile.id, revision=profile.revision)
    assert (await monitor.delete_profile(1, profile.id, revision=profile.revision)) == {'deleted': True}
    assert (await monitor.list_profiles(1)) == []
    assert (await monitor.list_events(1)).total == 0
    reader = monitor
    for read in (lambda: reader.read_profile(1, profile.id), lambda: reader.read_run(1, profile.id, run.id)):
        with pytest.raises(ProfileNotFound):
            await read()
    async with db.get_db_session() as session, session.begin():
        assert (await session.execute(text('SELECT count(*) AS n FROM intelligence.density_run_input WHERE run_id=:p0'), {'p0': run.id})).mappings().first()['n'] == 0
        assert (await session.execute(text('SELECT count(*) AS n FROM intelligence.density_event_state WHERE profile_id=:p0'), {'p0': profile.id})).mappings().first()['n'] == 0
    assert (await monitor.read_input(2, other.id, other_run.id, source.product)) == source
    assert (await monitor.list_events(2)).total > 0


async def test_ownership_revision_deduplication_and_immutable_event_source(storage,monkeypatch):
    p=await monitor.save_profile(1,body())
    assert (await monitor.list_profiles(2))==[]
    with pytest.raises(ProfileNotFound): await monitor.history(2,p.id)
    inputs={'atmospheric-density':release()}
    run=await monitor.assess_profile(1,p.id,inputs=inputs)
    assert (await monitor.assess_profile(1,p.id,inputs=inputs)).id==run.id
    assert len((await monitor.history(1,p.id)))==1
    assert (await monitor.list_events(2)).total==0
    events=await monitor.list_events(1)
    assert events.total==1 and events.items[0].run_id==run.id
    with pytest.raises(ProfileNotFound): await monitor.read_run(2,p.id,run.id)
    with pytest.raises(ProfileNotFound): await monitor.acknowledge(2,events.items[0].id)
    first=await monitor.acknowledge(1,events.items[0].id)
    assert (await monitor.acknowledge(1,first.id)).acknowledged_at==first.acknowledged_at
    changed=body().model_copy(update={'name':'New name'})
    updated=await monitor.save_profile(1,changed,profile_id=p.id,revision=1)
    assert updated.revision==2
    assert (await monitor.list_profiles(1))[0].data_status=='profile_changed'
    assert (await monitor.read_run(1,p.id,run.id)).profile_snapshot.name=='Supplied satellite'
    with pytest.raises(ProfileConflict): await monitor.save_profile(1,changed,profile_id=p.id,revision=1)
    with pytest.raises(ProfileNotFound): await monitor.save_profile(2,changed,profile_id=p.id,revision=2)


async def test_retained_inputs_survive_upstream_outage_and_are_shared_between_runs(storage, monkeypatch):
    source = release()
    inputs = {source.product: source, 'solar-wind-events': None}
    p = await monitor.save_profile(1, body())
    other = await monitor.save_profile(1, body())
    run = await monitor.assess_profile(1, p.id, inputs=inputs)
    second = await monitor.assess_profile(1, other.id, inputs=inputs)
    def unavailable(*a, **k):
        raise AssertionError('Stored input reads must not contact Prophet')
    monkeypatch.setattr(monitor, 'fetch_release', unavailable)
    assert (await monitor.read_input(1, p.id, run.id, source.product)) == source
    assert (await monitor.read_input(1, other.id, second.id, source.product)) == source
    async with db.get_db_session() as session, session.begin():
        assert (await session.execute(text('SELECT count(*) AS n FROM intelligence.forecast_input'))).mappings().first()['n'] == 1
        assert (await session.execute(text('SELECT count(*) AS n FROM intelligence.density_run_input'))).mappings().first()['n'] == 2
    with pytest.raises(ProfileNotFound): await monitor.read_input(2, p.id, run.id, source.product)
    with pytest.raises(ProfileNotFound): await monitor.read_input(1, other.id, run.id, source.product)
    with pytest.raises(ProfileNotFound): await monitor.read_input(1, p.id, run.id, 'solar-wind-events')


async def test_changed_release_id_rolls_back_assessment(storage):
    source = release()
    p = await monitor.save_profile(1, body())
    await monitor.assess_profile(1, p.id, inputs={source.product: source})
    other = await monitor.save_profile(1, body())
    changed = source.model_copy(update={'published_at': source.published_at + timedelta(seconds=1)})
    with pytest.raises(ValueError, match='same ID'):
        await monitor.assess_profile(1, other.id, inputs={source.product: changed})
    assert (await monitor.history(1, other.id)) == []
    assert (await monitor.list_events(1)).total == 1


async def test_retained_payload_integrity_is_checked_before_read(storage):
    import gzip
    source = release()
    p = await monitor.save_profile(1, body())
    run = await monitor.assess_profile(1, p.id, inputs={source.product: source})
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('UPDATE intelligence.forecast_input SET content_gzip=:p0 WHERE release_id=:p1'), {'p0': gzip.compress(b'{}'), 'p1': source.release_id})
    with pytest.raises(ValueError, match='corrupt'):
        await monitor.read_input(1, p.id, run.id, source.product)


async def test_journal_failure_rolls_back_input_copy(storage, monkeypatch):
    p = await monitor.save_profile(1, body())
    source = release()
    def fail(*a, **k): raise RuntimeError('Simulated journal failure')
    monkeypatch.setattr(monitor, 'record_events', fail)
    with pytest.raises(RuntimeError): await monitor.assess_profile(1, p.id, inputs={source.product: source})
    async with db.get_db_session() as session, session.begin():
        assert (await session.execute(text('SELECT count(*) AS n FROM intelligence.forecast_input'))).mappings().first()['n'] == 0
    assert (await monitor.history(1, p.id)) == []


async def test_revision_race_rolls_back_run_and_journal(storage,monkeypatch):
    p=await monitor.save_profile(1,body())
    loop = asyncio.get_running_loop()
    def race(profile,inputs,now):
        asyncio.run_coroutine_threadsafe(monitor.save_profile(1,body(),profile_id=p.id,revision=1), loop).result()
        return canned(profile,inputs,now)
    monkeypatch.setattr(monitor,'prepare_assessment',race)
    with pytest.raises(ProfileConflict): await monitor.assess_profile(1,p.id,inputs={})
    assert (await monitor.history(1,p.id))==[]
    assert (await monitor.list_events(1)).total==0


async def test_journal_failure_cannot_publish_partial_run(storage,monkeypatch):
    p=await monitor.save_profile(1,body())
    def fail(*a,**k): raise RuntimeError('Simulated journal write failure')
    monkeypatch.setattr(monitor,'record_events',fail)
    with pytest.raises(RuntimeError): await monitor.assess_profile(1,p.id,inputs={})
    assert (await monitor.history(1,p.id))==[]


pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
async def database_runtime(storage):
    try:
        yield
    finally:
        await dispose_engine()
