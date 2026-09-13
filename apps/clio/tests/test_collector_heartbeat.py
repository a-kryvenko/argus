import json
from argus_clio.services import collector_heartbeat as service


def test_missing_progress_file_is_unhealthy(tmp_path):
    assert not service.check_heartbeat('solar-wind', tmp_path/'missing')['healthy']


def test_initial_progress_and_clean_shutdown(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'monotonic', lambda: 100)
    path = tmp_path/'health.json'
    beat = service.CollectorHeartbeat('solar-wind', path)
    assert not service.check_heartbeat('solar-wind', path, now=100)['healthy']
    for source_id in beat.sources:
        beat.started(source_id); beat.finished(source_id)
    assert service.check_heartbeat('solar-wind', path, now=110)['healthy']
    beat.stop()
    assert service.check_heartbeat('solar-wind', path, now=111)['reason'] == 'collector_stopped'


def test_stalled_attempt_and_overdue_cycle_are_different(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'monotonic', lambda: 100)
    path = tmp_path/'health.json'; beat = service.CollectorHeartbeat('solar-wind', path)
    for source_id in beat.sources:
        beat.started(source_id)
    result = service.check_heartbeat('solar-wind', path, now=221)
    assert set(result['sources'].values()) == {'stalled'}
    for source_id in beat.sources:
        beat.finished(source_id)
    result = service.check_heartbeat('solar-wind', path, now=281)
    assert set(result['sources'].values()) == {'overdue'}


def test_dst_waiting_for_five_minute_cycle_is_healthy(monkeypatch, tmp_path):
    path = tmp_path/'health.json'; beat = service.CollectorHeartbeat('geomagnetic', path)
    monkeypatch.setattr(service, 'monotonic', lambda: 100)
    beat.started('dst'); beat.finished('dst')
    monkeypatch.setattr(service, 'monotonic', lambda: 350)
    beat.started('kp'); beat.finished('kp')
    assert service.check_heartbeat('geomagnetic', path, now=360)['healthy']


def test_invalid_or_other_collectors_heartbeat_cannot_mask_failure(tmp_path):
    path = tmp_path/'health.json'
    for payload in ['{broken', '{}', json.dumps({'collector':'geomagnetic','sources':{}}),
                    json.dumps({'collector':'solar-wind','active':True,'sources':{'solar_wind_mag':{'started':float('nan'), 'finished':None}}})]:
        path.write_text(payload)
        assert not service.check_heartbeat('solar-wind', path, now=100)['healthy']
