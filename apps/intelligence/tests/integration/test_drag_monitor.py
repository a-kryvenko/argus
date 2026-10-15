"""Ownership, version races, deduplication and durable failure history on PostgreSQL."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
import psycopg
from domain_storage import database, migrate
from common.schemas.drag_monitor import DragProfileInput
from common.schemas.leo_drag import LeoDragAssessment
from argus_intelligence import monitor
from argus_intelligence import events
from argus_intelligence.drag import DragDomainError


@pytest.fixture
def storage(database, monkeypatch):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        if key.startswith('INTELLIGENCE_DB_'):
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(monitor, 'get_config', lambda: SimpleNamespace(models_registry={'models': {'atmospheric_density': {'max_age_hours': 6}}}))
    monkeypatch.setattr(events, 'get_config', monitor.get_config)


def body():
    return DragProfileInput(name='Satellite A', altitude_km=400, inclination_deg=51.6, mass_kg=100,
        thresholds={'elevated_altitude_loss_m': 50, 'high_altitude_loss_m': 100},
        modes=[{'name': 'Nominal', 'effective_area_m2': 1, 'drag_coefficient': 2.2},
               {'name': 'Safe', 'effective_area_m2': 2, 'drag_coefficient': 2.2}], active_mode='Nominal')


def assessment(request, *, release):
    now = datetime.now(UTC)
    points = [dict(valid_time=now+timedelta(hours=i), lead_hours=i, mean_density_kg_m3=1e-12,
        mean_drag_accel_m_s2=1e-6, mean_along_track_deceleration_m_s2=1e-6,
        delta_v_loss_m_s=i / 240, estimated_altitude_loss_m=100*i / 24) for i in range(25)]
    return LeoDragAssessment(computed_at=now, start_time=now, end_time=now+timedelta(hours=24),
        inputs=request, source=dict(release_id=release.release_id, issue_time=now, observed_at=now,
            dtc_observed_at=now, history_start=now-timedelta(days=81),
            background_method='trailing_81_daily_values', background_interpolated_days={}, dtc_method='causal_dst_ap_v1'),
        mean_density_kg_m3=1e-12, mean_drag_accel_m_s2=1e-6, delta_v_loss_m_s=0.1,
        estimated_altitude_loss_m=100, drag_risk='high', risk_reason='User threshold reached',
        assumptions=['Circular orbit'], predictions=points)


def test_owner_isolation_versions_and_durable_history(storage, monkeypatch):
    profile = monitor.save_profile(1, body())
    assert monitor.list_profiles(2) == []
    with pytest.raises(monitor.ProfileNotFound):
        monitor.history(2, profile.id)
    with pytest.raises(monitor.ProfileNotFound):
        monitor.save_profile(2, body(), profile_id=profile.id, revision=1)
    release = SimpleNamespace(release_id=uuid4())
    monkeypatch.setattr(monitor, 'assess_drag', assessment)
    run = monitor.assess_profile(1, profile.id, release=release)
    assert len(run.modes) == 2
    assert run.modes[0].elevated_at is not None
    assert run.modes[0].high_at == run.modes[0].assessment.end_time
    monkeypatch.setattr(monitor, 'assess_drag', lambda *a, **kw: pytest.fail('Duplicate computation'))
    assert monitor.assess_profile(1, profile.id, release=release).id == run.id
    assert len(monitor.history(1, profile.id)) == 1
    changed = body().model_copy(update={'mass_kg': 200})
    new = monitor.save_profile(1, changed, profile_id=profile.id, revision=1)
    assert new.revision == 2
    assert monitor.list_profiles(1)[0].data_status == 'profile_changed'
    assert monitor.history(1, profile.id)[0].profile_snapshot.mass_kg == 100
    with pytest.raises(monitor.ProfileConflict):
        monitor.save_profile(1, body(), profile_id=profile.id, revision=1)


def test_failure_supersedes_success_and_does_not_block_other_modes(storage, monkeypatch):
    profile = monitor.save_profile(1, body())
    monkeypatch.setattr(monitor, 'assess_drag', assessment)
    monitor.assess_profile(1, profile.id, release=SimpleNamespace(release_id=uuid4()))
    failed = monitor.assess_profile(1, profile.id, release=None)
    assert all(mode.error == 'unavailable' for mode in failed.modes)
    assert monitor.list_profiles(1)[0].attention == 'not_assessed'
    assert len(monitor.history(1, profile.id)) == 2
    def partial(request, *, release):
        if request.effective_area_m2 == 2:
            raise DragDomainError('private backend information')
        return assessment(request, release=release)
    monkeypatch.setattr(monitor, 'assess_drag', partial)
    partial_run = monitor.assess_profile(1, profile.id, release=SimpleNamespace(release_id=uuid4()))
    assert partial_run.modes[0].assessment is not None
    assert partial_run.modes[1].error == 'outside_model_domain'
    assert 'private backend' not in partial_run.model_dump_json()


def test_edit_during_calculation_cannot_publish_and_lock_excludes_writer(storage, monkeypatch):
    profile = monitor.save_profile(1, body())
    with monitor.db.connect() as conn:
        conn.execute('SELECT pg_advisory_lock(hashtextextended(%s, 0))', (str(profile.id),))
        with pytest.raises(monitor.ProfileConflict):
            monitor.assess_profile(1, profile.id, release=None)
    edited = False
    def racing(request, *, release):
        nonlocal edited
        if not edited:
            monitor.save_profile(1, body(), profile_id=profile.id, revision=1)
            edited = True
        return assessment(request, release=release)
    monkeypatch.setattr(monitor, 'assess_drag', racing)
    with pytest.raises(monitor.ProfileConflict):
        monitor.assess_profile(1, profile.id, release=SimpleNamespace(release_id=uuid4()))
    assert monitor.history(1, profile.id) == []


def test_worker_new_release_retry_pause_and_shared_fetch(storage, monkeypatch):
    profile = monitor.save_profile(1, body())
    release = SimpleNamespace(release_id=uuid4())
    calls = []
    monkeypatch.setattr(monitor, 'fetch_release', lambda _: calls.append(1) or release)
    monkeypatch.setattr(monitor, 'assess_drag', assessment)
    assert monitor.refresh_profiles() == {'profiles': 1, 'failed': 0}
    assert monitor.refresh_profiles() == {'profiles': 1, 'failed': 0}
    assert len(monitor.history(1, profile.id)) == 1
    release = SimpleNamespace(release_id=uuid4())
    monitor.refresh_profiles()
    assert len(monitor.history(1, profile.id)) == 2
    paused = body().model_copy(update={'enabled': False})
    monitor.save_profile(1, paused, profile_id=profile.id, revision=1)
    assert monitor.refresh_profiles()['profiles'] == 0
    with pytest.raises(monitor.ProfileConflict):
        monitor.assess_profile(1, profile.id, release=release)
    assert len(calls) == 3


def test_operator_chart_selects_owned_snapshot_and_delegates_to_private_core(storage, monkeypatch):
    import intelligence_core.api as backend
    profile = monitor.save_profile(1, body())
    other = monitor.save_profile(2, body())
    with pytest.raises(monitor.ProfileNotFound):
        monitor.operator_chart(1, profile.id)
    monkeypatch.setattr(monitor, 'assess_drag', assessment)
    first = monitor.assess_profile(1, profile.id, release=SimpleNamespace(release_id=uuid4()))
    second = monitor.assess_profile(1, profile.id, release=SimpleNamespace(release_id=uuid4()))
    foreign = monitor.assess_profile(2, other.id, release=SimpleNamespace(release_id=uuid4()))
    calls = []
    sentinel = object()
    def prepare(profile, run, prior, **kwargs):
        calls.append((profile, run, prior, kwargs))
        return sentinel
    monkeypatch.setattr(backend, 'prepare_operator_chart', prepare)
    assert monitor.operator_chart(1, profile.id) is sentinel
    assert calls[-1][1].id == second.id
    assert [item.id for item in calls[-1][2]] == [first.id]
    assert monitor.operator_chart(1, profile.id, run_id=first.id) is sentinel
    assert calls[-1][1].id == first.id
    assert calls[-1][2] == []
    assert calls[-1][3]['latest_run_id'] == second.id
    with pytest.raises(monitor.ProfileNotFound):
        monitor.operator_chart(2, profile.id)
    with pytest.raises(monitor.ProfileNotFound):
        monitor.operator_chart(1, profile.id, run_id=foreign.id)


def test_event_delivery_owner_isolation_idempotent_review_and_exact_run(storage, monkeypatch):
    monkeypatch.setattr(monitor, 'assess_drag', assessment)
    profile = monitor.save_profile(1, body())
    other = monitor.save_profile(2, body())
    saved = monitor.assess_profile(1, profile.id, release=SimpleNamespace(release_id=uuid4()))
    monitor.assess_profile(2, other.id, release=SimpleNamespace(release_id=uuid4()))
    feed = events.list_events(1)
    assert feed.total == feed.unacknowledged == 1
    event = feed.items[0]
    assert event.profile_id == profile.id
    assert event.run_id == saved.id
    events.process_events(1, profile.id)
    assert events.list_events(1).total == 1
    with pytest.raises(monitor.ProfileNotFound):
        events.acknowledge(2, event.id)
    with pytest.raises(monitor.ProfileNotFound):
        events.list_events(2, profile_id=profile.id)
    reviewed = events.acknowledge(1, event.id)
    assert reviewed.acknowledged_by == 1
    assert events.acknowledge(1, event.id).acknowledged_at == reviewed.acknowledged_at
    assert events.list_events(1, unacknowledged=True).total == 0
    assert monitor.read_run(1, profile.id, saved.id).id == saved.id
    with pytest.raises(monitor.ProfileNotFound):
        monitor.read_run(2, profile.id, saved.id)


def test_event_failure_rolls_back_assessment_and_transition_state(storage, monkeypatch):
    monkeypatch.setattr(monitor, 'assess_drag', assessment)
    profile = monitor.save_profile(1, body())
    with monitor.db.connect() as conn:
        conn.execute('''CREATE FUNCTION intelligence.reject_event() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'forced event write failure'; END $$''')
        conn.execute('CREATE TRIGGER reject_event BEFORE INSERT ON intelligence.drag_event FOR EACH ROW EXECUTE FUNCTION intelligence.reject_event()')
    with pytest.raises(psycopg.Error):
        monitor.assess_profile(1, profile.id, release=SimpleNamespace(release_id=uuid4()))
    assert monitor.history(1, profile.id) == []
    assert events.list_events(1).total == 0
    with monitor.db.connect() as conn:
        assert conn.execute('SELECT count(*) AS n FROM intelligence.drag_event_state').fetchone()['n'] == 0
        conn.execute('DROP TRIGGER reject_event ON intelligence.drag_event')
        conn.execute('DROP FUNCTION intelligence.reject_event()')
    monitor.assess_profile(1, profile.id, release=SimpleNamespace(release_id=uuid4()))
    assert events.list_events(1).total == 1
