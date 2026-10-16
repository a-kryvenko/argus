"""Composite demo publication is all-or-nothing across scientific stages."""
import copy
from datetime import UTC, datetime, timedelta
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('demo_pipeline', ROOT / 'scripts/demo/rebuild.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def content():
    from argus_prophet.demo import Manifest, issue_times
    manifest = Manifest.model_validate(dict(event=dict(name='Test', starts_at='2026-01-19T19:38Z',
        source_url='https://example.com/event', description='Synthetic test'), products=['solar-wind-density'],
        observation_source='Test', input_source='Test', models={'plasma_density_quantile':dict(
        sha256='a'*64, training_end='2025-01-01T00:00Z', training_evidence='Test')}))
    times = issue_times(manifest.event)
    bundle = manifest.model_dump(mode='json', exclude={'products'}) | dict(schema_version=1, version='pending',
        releases={'solar-wind-density':[dict(issue_time=t.isoformat()) for t in times]})
    scenario = dict(event=bundle['event'], releases={v:[dict(issue_time=t.isoformat(),
        predictions=[dict(time=(t+timedelta(hours=h)).isoformat()) for h in range(1,49)]) for t in times]
        for v in ('jb2008_fixed','jb2008_forecast','dtm2020_fixed','dtm2020_forecast')})
    return bundle, scenario


def setup(tmp_path, monkeypatch, fail=None):
    args = SimpleNamespace(scenario='january-2026', refresh_data=False)
    raw, grace = tmp_path/'raw', tmp_path/'grace'
    (raw/'goes').mkdir(parents=True)
    grace.mkdir()
    for day in range(12,25):
        stamp=f'202601{day:02d}'
        (grace/f'GF_OPER_DNS1ACC_2__{stamp}T000000_{stamp}T235959_0003.cdf').write_bytes(b'cache')
    archives=[]
    for year in (2025,2026):
        p=raw/f'omni2_{year}.dat'; p.write_bytes(f'forecast archive {year}'.encode()); archives.append(p)
    destination=tmp_path/'data/demo'; destination.mkdir(parents=True)
    current=destination/'current.json'; current.write_bytes(b'{"version":"previous","density_scenario":{"retained":true}}')
    old=current.read_bytes()
    bundle, scenario=content()
    monkeypatch.setattr(pipeline,'preflight',lambda *args:dict(raw=raw, grace=grace,dtm=tmp_path/'dtm',manifest=tmp_path/'manifest'))
    calls=[]
    def forecast(args, pending):
        assert current.read_bytes()==old
        calls.append('forecast')
        if fail=='forecast': raise ValueError('inference failed')
        pending.parent.mkdir(parents=True)
        pending.write_text(json.dumps(bundle))
        return archives
    def stage(command, root):
        assert current.read_bytes()==old
        name=Path(command[0]).name; calls.append(name)
        if fail==name: raise subprocess.CalledProcessError(1, name)
        run=next(destination.joinpath('runs').iterdir())
        if name=='run_event.py':
            view=Path(command[command.index('--raw-root')+1])
            for archive in archives: assert (view/'omni_hourly'/archive.name).read_bytes()==archive.read_bytes()
        if name=='density_event_replay.py':
            output=run/'density/four-way'; output.mkdir(parents=True)
            out=copy.deepcopy(scenario)
            if fail=='validation': out['releases']['jb2008_fixed'].pop()
            (output/'scenario.json').write_text(json.dumps(out))
    monkeypatch.setattr(pipeline,'forecast_stage',forecast)
    monkeypatch.setattr(pipeline,'stage',stage)
    return args,current,old,calls


@pytest.mark.parametrize('failure',['forecast','run_event.py','run_event_forecast.py','density_event_replay.py','validation'])
def test_failed_stage_retains_previous_complete_demo(tmp_path,monkeypatch,failure):
    args,current,previous,calls=setup(tmp_path,monkeypatch,fail=failure)
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        pipeline.rebuild(args,tmp_path)
    assert current.read_bytes()==previous


def test_all_stages_publish_together_and_use_same_forecast_archives(tmp_path,monkeypatch):
    args,current,previous,calls=setup(tmp_path,monkeypatch)
    result=pipeline.rebuild(args,tmp_path)
    assert calls==['forecast','run_event.py','run_event_forecast.py','density_event_replay.py']
    published=json.loads(current.read_text())
    assert published==result
    assert 'density_scenario' in published and published['version']!='previous'
    assert json.loads((current.parent/(published['version']+'.json')).read_text())==published


def test_concurrent_build_does_not_start_stages(tmp_path,monkeypatch):
    args,current,previous,calls=setup(tmp_path,monkeypatch)
    import fcntl
    with (current.parent/'.build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='Another demo build'):
            pipeline.rebuild(args,tmp_path)
    assert not calls and current.read_bytes()==previous


def test_invalid_leo_hour_cannot_publish():
    bundle,scenario=content()
    bundle['density_scenario']=scenario
    scenario['releases']['dtm2020_forecast'][0]['predictions'][0]['time']='2026-01-01T00:00Z'
    with pytest.raises(ValueError,match='horizon'):
        pipeline.validate_complete(bundle)


def test_missing_product_cannot_publish():
    bundle,scenario=content()
    bundle['density_scenario']=scenario
    with pytest.raises(ValueError,match='product set'):
        pipeline.validate_complete(bundle,['solar-wind-density','dst'])
