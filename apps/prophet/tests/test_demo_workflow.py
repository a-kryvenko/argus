import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from argus_prophet.commands import demo
from argus_prophet.demo_sources import normalized_points


def test_inputs_exclude_future_rows_and_unclosed_indices():
    issue = pd.Timestamp('2026-01-19T19:00Z')
    times = pd.date_range(issue-pd.Timedelta(days=10), issue+pd.Timedelta(hours=5), freq='h')
    frame = pd.DataFrame({'issue_time': times, **{key: 1. for key in ['bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'ap', 'dst', 'f10_7']}})
    # These values are in the archive, but not known at simulated 19:00.
    frame.loc[frame.issue_time >= issue.floor('D'), 'f10_7'] = 999
    frame.loc[frame.issue_time >= issue.floor('3h'), ['kp', 'ap']] = 9
    frame.loc[frame.issue_time >= issue, 'v'] = 2000
    result = normalized_points(frame, issue)
    assert result[-1]['f10_7'] == 1
    assert result[-1]['kp'] == 1
    assert result[-1]['ap'] == 1
    assert result[-1]['v'] == 1
    assert pd.Timestamp(result[-1]['issue_time']) == issue-pd.Timedelta(hours=1)


def test_cache_hit_does_not_download(tmp_path, monkeypatch):
    from common.data import omni
    config = SimpleNamespace(data_root=tmp_path)
    folder = tmp_path / 'cache'
    folder.mkdir()
    destination = folder / 'omni2_2026.dat'
    destination.write_text('cached')
    monkeypatch.setattr(omni, 'parse_annual', lambda text, year: None)
    import httpx
    monkeypatch.setattr(httpx, 'get', lambda *args, **kwargs: pytest.fail('cache hit must not access network'))
    assert demo.archive(2026, config, folder) == destination


def test_failed_refresh_keeps_cached_archive(tmp_path, monkeypatch):
    from common.data import omni
    import httpx
    destination = tmp_path / 'omni2_2026.dat'
    destination.write_text('old')
    monkeypatch.setattr(httpx, 'get', lambda *args, **kwargs: SimpleNamespace(content=b'bad', raise_for_status=lambda: None))
    def reject(*args):
        raise ValueError('Invalid year')
    monkeypatch.setattr(omni, 'parse_annual', reject)
    with pytest.raises(ValueError, match='Invalid year'):
        demo.archive(2026, SimpleNamespace(data_root=tmp_path), tmp_path, refresh=True)
    assert destination.read_text() == 'old'


def test_complete_command_prepares_builds_and_preserves_previous_on_failure(tmp_path, monkeypatch):
    from common import config as config_module
    from argus_prophet import demo as builder, demo_sources
    config = SimpleNamespace(data_root=tmp_path / 'data', config_root=tmp_path / 'configs',
                             models_registry={'models': {'plasma_density_quantile': {'model': 'density'}}})
    model = config.data_root / 'models/density.joblib'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'model')
    manifest = {'event': {'name': 'Test', 'starts_at': '2026-01-19T19:38Z', 'source_url': 'https://example.com', 'description': 'Test'},
                'products': ['solar-wind-density'], 'observation_source': 'test', 'input_source': 'test',
                'models': {'plasma_density_quantile': {'sha256': hashlib.sha256(b'model').hexdigest(),
                           'training_end': '2025-01-01T00:00Z', 'training_evidence': 'test'}}}
    path = config.config_root / 'demo/january-2026.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    current = config.data_root / 'demo/current.json'
    current.parent.mkdir()
    current.write_text('previous')
    monkeypatch.setattr(config_module, 'get_config', lambda: config)
    downloads = []
    def archive(year, config, folder, refresh):
        downloads.append((year, refresh))
        return tmp_path / str(year)
    monkeypatch.setattr(demo, 'archive', archive)
    def prepare(path, archives, destination):
        destination.mkdir(parents=True)
        (destination / 'observations.json').write_text('[]')
    monkeypatch.setattr(demo_sources, 'prepare', prepare)
    def fail(*args):
        raise ValueError('Incomplete model output')
    monkeypatch.setattr(builder, 'build', fail)
    with pytest.raises(ValueError, match='previous published dataset preserved'):
        demo.run(SimpleNamespace(scenario='january-2026', refresh_data=True))
    assert downloads == [(2025, True), (2026, True)]
    assert current.read_text() == 'previous'


def test_path_traversal_is_rejected_before_loading_config():
    with pytest.raises(ValueError, match='Scenario must be a name'):
        demo.run(SimpleNamespace(scenario='../private', refresh_data=False))
