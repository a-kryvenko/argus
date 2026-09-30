from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import numpy as np
import requests

from clio.commands import sdo_images as collector
from clio.providers.sdo_images import observation_url, prepare_observation
from common.sdo_images import hourly_slots, read_image

SLOT = datetime(2026, 9, 30, 6, tzinfo=UTC)


def test_urls_and_reduction_preserve_missing_measurements_and_wcs():
    assert observation_url('aia1600', SLOT).endswith('/H0600/AIA20260930_060000_1600.fits')
    assert observation_url('hmi_m', SLOT).endswith('hmi.M_720s_nrt.20260930_060000_TAI.fits')
    image = np.full((1024, 1024), 4, np.float32)
    image[:2, :2] = np.nan
    image[2, 2] = np.nan
    reduced, meta = prepare_observation(image, dict(channel='hmi_m', header={
        'CRPIX1': 512.5, 'CDELT1': 2., 'HISTORY': ['source']}))
    assert np.isnan(reduced[0, 0]) and reduced[1, 1] == 4
    assert meta['header']['CRPIX1'] == 256.5 and meta['header']['CDELT1'] == 4
    assert meta['source_header']['CRPIX1'] == 512.5
    assert meta['units'] == 'Gauss'


def test_live_publishes_once_and_cleanup_expires_files(tmp_path, monkeypatch):
    now = SLOT + timedelta(minutes=30)
    def download(url, channel, *, slot, now):
        return np.ones((1024, 1024), np.float32), dict(
            channel=channel, header={}, source=url, sha256='digest',
            observed_at=slot.isoformat(), available_at=now.isoformat())
    download = Mock(side_effect=download)
    monkeypatch.setattr(collector, 'download_observation', download)
    report = collector.cycle(tmp_path, 'live', clock=lambda: now)
    assert report['saved'] == 36
    array, metadata = read_image(tmp_path, SLOT, 'aia1600')
    assert array.shape == (512, 512) and metadata['units'] == 'DN'
    assert collector.cycle(tmp_path, 'live', clock=lambda: now)['existing'] == 36
    assert download.call_count == 36
    assert collector.cycle(tmp_path, 'cleanup', clock=lambda: now + timedelta(hours=144))['removed'] == 36


def test_warmup_visits_entire_window_despite_missing_sources(tmp_path, monkeypatch):
    now = SLOT + timedelta(minutes=30)
    visited = set()
    def missing(url, channel, *, slot, now):
        visited.add(slot)
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError(response=response)
    monkeypatch.setattr(collector, 'download_observation', missing)
    for _ in range(24):
        report = collector.cycle(tmp_path, 'warmup', clock=lambda: now)
        assert report['unavailable'] == 54
    assert visited == set(hourly_slots(now)[4:])
    assert not list(tmp_path.glob('*/*.npz'))


def test_worker_registers_independent_sdo_lanes(monkeypatch):
    from clio import worker
    from clio.config import ClioObservations
    calls = []
    monkeypatch.setattr(worker, 'invoke_isolated', lambda name, args, **kw: calls.append((name, args.mode)))
    config = ClioObservations(observations={}, sdo_images={'enabled': True})
    tasks = {t.name: t for t in worker.tasks_for(config, {}, None)}
    for mode in ('live', 'warmup', 'cleanup'):
        task = tasks[f'sdo-{mode}']
        assert task.next_run == 0 and task.background == (mode == 'warmup')
        task.run()
    assert calls == [('sdo-images', mode) for mode in ('live', 'warmup', 'cleanup')]
    config.sdo_images.enabled = False
    assert not any(t.name.startswith('sdo-') for t in worker.tasks_for(config, {}, None))


def test_lane_lock_skips_duplicate_worker_but_allows_cleanup(tmp_path):
    import fcntl
    with (tmp_path / '.warmup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert collector.cycle(tmp_path, 'warmup') == {'busy': True}
        assert collector.cycle(tmp_path, 'cleanup') == {'removed': 0}


def test_failed_download_does_not_publish_or_block_other_channels(tmp_path, monkeypatch):
    now = SLOT + timedelta(minutes=30)
    def fail(url, channel, **kwargs):
        raise requests.Timeout('upstream timeout')
    monkeypatch.setattr(collector, 'download_observation', fail)
    assert collector.cycle(tmp_path, 'live', clock=lambda: now)['failed'] == 36
    assert not list(tmp_path.glob('*/*.npz'))
