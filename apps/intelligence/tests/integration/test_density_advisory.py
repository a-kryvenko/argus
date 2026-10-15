"""Real PostgreSQL ownership, revision races, immutable snapshots and event storage."""
from datetime import UTC,datetime,timedelta
import hashlib
from uuid import uuid4

import pytest
from domain_storage import database,migrate
from common.schemas.density_advisory import AdvisoryProfileInput,DensityAdvisory
from common.schemas.forecast_release import ForecastRelease
from argus_intelligence import density_monitor as monitor
from argus_intelligence.monitor import ProfileConflict,ProfileNotFound


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


def test_ownership_revision_deduplication_and_immutable_event_source(storage,monkeypatch):
    p=monitor.save_profile(1,body())
    assert monitor.list_profiles(2)==[]
    with pytest.raises(ProfileNotFound): monitor.history(2,p.id)
    inputs={'atmospheric-density':release()}
    run=monitor.assess_profile(1,p.id,inputs=inputs)
    assert monitor.assess_profile(1,p.id,inputs=inputs).id==run.id
    assert len(monitor.history(1,p.id))==1
    assert monitor.list_events(2).total==0
    events=monitor.list_events(1)
    assert events.total==1 and events.items[0].run_id==run.id
    with pytest.raises(ProfileNotFound): monitor.read_run(2,p.id,run.id)
    with pytest.raises(ProfileNotFound): monitor.acknowledge(2,events.items[0].id)
    first=monitor.acknowledge(1,events.items[0].id)
    assert monitor.acknowledge(1,first.id).acknowledged_at==first.acknowledged_at
    changed=body().model_copy(update={'name':'New name'})
    updated=monitor.save_profile(1,changed,profile_id=p.id,revision=1)
    assert updated.revision==2
    assert monitor.list_profiles(1)[0].data_status=='profile_changed'
    assert monitor.read_run(1,p.id,run.id).profile_snapshot.name=='Supplied satellite'
    with pytest.raises(ProfileConflict): monitor.save_profile(1,changed,profile_id=p.id,revision=1)
    with pytest.raises(ProfileNotFound): monitor.save_profile(2,changed,profile_id=p.id,revision=2)


def test_retained_inputs_survive_upstream_outage_and_are_shared_between_runs(storage, monkeypatch):
    source = release()
    inputs = {source.product: source, 'solar-wind-events': None}
    p = monitor.save_profile(1, body())
    other = monitor.save_profile(1, body())
    run = monitor.assess_profile(1, p.id, inputs=inputs)
    second = monitor.assess_profile(1, other.id, inputs=inputs)
    def unavailable(*a, **k):
        raise AssertionError('Stored input reads must not contact Prophet')
    monkeypatch.setattr(monitor, 'fetch_release', unavailable)
    assert monitor.read_input(1, p.id, run.id, source.product) == source
    assert monitor.read_input(1, other.id, second.id, source.product) == source
    with monitor.db.connect() as conn:
        assert conn.execute('SELECT count(*) AS n FROM intelligence.forecast_input').fetchone()['n'] == 1
        assert conn.execute('SELECT count(*) AS n FROM intelligence.density_run_input').fetchone()['n'] == 2
    with pytest.raises(ProfileNotFound): monitor.read_input(2, p.id, run.id, source.product)
    with pytest.raises(ProfileNotFound): monitor.read_input(1, other.id, run.id, source.product)
    with pytest.raises(ProfileNotFound): monitor.read_input(1, p.id, run.id, 'solar-wind-events')


def test_changed_release_id_rolls_back_assessment(storage):
    source = release()
    p = monitor.save_profile(1, body())
    monitor.assess_profile(1, p.id, inputs={source.product: source})
    other = monitor.save_profile(1, body())
    changed = source.model_copy(update={'published_at': source.published_at + timedelta(seconds=1)})
    with pytest.raises(ValueError, match='same ID'):
        monitor.assess_profile(1, other.id, inputs={source.product: changed})
    assert monitor.history(1, other.id) == []
    assert monitor.list_events(1).total == 1


def test_retained_payload_integrity_is_checked_before_read(storage):
    import gzip
    source = release()
    p = monitor.save_profile(1, body())
    run = monitor.assess_profile(1, p.id, inputs={source.product: source})
    with monitor.db.connect() as conn:
        conn.execute('UPDATE intelligence.forecast_input SET content_gzip=%s WHERE release_id=%s',
                     (gzip.compress(b'{}'), source.release_id))
    with pytest.raises(ValueError, match='corrupt'):
        monitor.read_input(1, p.id, run.id, source.product)


def test_journal_failure_rolls_back_input_copy(storage, monkeypatch):
    p = monitor.save_profile(1, body())
    source = release()
    def fail(*a, **k): raise RuntimeError('Simulated journal failure')
    monkeypatch.setattr(monitor, 'record_events', fail)
    with pytest.raises(RuntimeError): monitor.assess_profile(1, p.id, inputs={source.product: source})
    with monitor.db.connect() as conn:
        assert conn.execute('SELECT count(*) AS n FROM intelligence.forecast_input').fetchone()['n'] == 0
    assert monitor.history(1, p.id) == []


def test_revision_race_rolls_back_run_and_journal(storage,monkeypatch):
    p=monitor.save_profile(1,body())
    def race(profile,inputs,now):
        monitor.save_profile(1,body(),profile_id=p.id,revision=1)
        return canned(profile,inputs,now)
    monkeypatch.setattr(monitor,'prepare_assessment',race)
    with pytest.raises(ProfileConflict): monitor.assess_profile(1,p.id,inputs={})
    assert monitor.history(1,p.id)==[]
    assert monitor.list_events(1).total==0


def test_journal_failure_cannot_publish_partial_run(storage,monkeypatch):
    p=monitor.save_profile(1,body())
    def fail(*a,**k): raise RuntimeError('Simulated journal write failure')
    monkeypatch.setattr(monitor,'record_events',fail)
    with pytest.raises(RuntimeError): monitor.assess_profile(1,p.id,inputs={})
    assert monitor.history(1,p.id)==[]
