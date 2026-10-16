"""Rebuild forecasts and the LEO scenario; publish the complete bundle atomically."""
import argparse
from datetime import UTC, datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def environment(root):
    paths = [root, root / 'apps/prophet/src', root / 'apps/clio/src',
             root / 'packages/common/src', root / 'packages/forecast/src',
             root / 'packages/forecast-core/src', root / 'packages/intelligence-core/src']
    return {**os.environ, 'ARGUS_WORKDIR': str(root),
            'PYTHONPATH': os.pathsep.join(map(str, paths)),
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'MPLCONFIGDIR': str(root / 'data/demo/.matplotlib')}


def preflight(args, root):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', args.scenario):
        raise ValueError('Expected a scenario name such as january-2026')
    # The research adapters currently pin the January 2026 event and GOES versions.
    if args.scenario != 'january-2026':
        raise ValueError('The complete LEO pipeline currently supports january-2026 only')
    raw = args.raw_root or (root / 'data/mnt/raw' if (root / 'data/mnt/raw/omni_hourly').is_dir() else root / 'data/raw')
    grace = args.grace_root or root / 'data/experiments/january-2026-density/gracefo'
    dtm = args.dtm_runtime or root / 'data/experiments/dtm2020_jb2008_v1'
    required = [root / 'configs/demo/january-2026.json',
        root / 'data/demo/january-2026/proswin-science-hourly.json',
        root / 'notebooks/workflows/training/thermosphere/run_event.py',
        root / 'notebooks/workflows/training/thermosphere/run_event_forecast.py',
        root / 'packages/intelligence-core/tools/density_event_replay.py',
        dtm / 'libdtm2020.so', dtm / 'build.json', dtm / 'download_manifest.json',
        dtm / 'upstream/DTM_2020_F107_Kp.dat',
        raw / 'goes/archive/sci_euvs-l2-avg1m_g18_y2025_v1-0-7.nc',
        raw / 'goes/archive/sci_euvs-l2-avg1m_g18_y2026_v1-0-7.nc',
        raw / 'goes/archive/sci_xrsf-l2-bkd1d_g18_s20220617_e20261011_v2-2-1.nc']
    for day in range(12, 25):
        stamp = f'202601{day:02d}'
        required.append(grace / f'GF_OPER_DNS1ACC_2__{stamp}T000000_{stamp}T235959_0003.cdf')
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError('Missing demo research inputs:\n' + '\n'.join(missing))
    for name in ('argus_prophet', 'clio', 'forecast_core', 'intelligence_core', 'cdflib', 'matplotlib'):
        if importlib.util.find_spec(name) is None:
            raise ValueError(f'Missing research dependency: {name}')
    from common.config import get_config
    from argus_prophet.demo import Manifest, validate_evidence, event_archive
    config = get_config()
    manifest = Manifest.model_validate_json(required[0].read_text())
    validate_evidence(manifest)
    if manifest.event_archive:
        event_archive(manifest, root)
    for name, evidence in manifest.models.items():
        entry = config.models_registry['models'][name]
        artifact = root / 'data/models' / (entry['model'] + '.joblib')
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != evidence.sha256:
            raise ValueError(f'{name}: update model hashes and training evidence before rebuilding')
    return dict(raw=raw.resolve(), grace=grace.resolve(), dtm=dtm.resolve(), manifest=required[0],
                products=manifest.products)


def forecast_stage(args, output):
    from argus_prophet.commands.demo import run
    archives = []
    run(SimpleNamespace(scenario=args.scenario, refresh_data=args.refresh_data), output=output,
        archive_paths=archives, raw_root=args.raw_root)
    return archives


def stage(command, root):
    subprocess.run([sys.executable, *map(str, command)], cwd=root, env=environment(root), check=True)


def validate_complete(bundle, expected_products=None):
    from argus_prophet.demo import Manifest, issue_times
    from datetime import datetime as dt
    spec = Manifest.model_validate({k: v for k, v in bundle.items()
        if k in Manifest.model_fields and k != 'products'} | {'products': list(bundle['releases'])})
    expected = issue_times(spec.event)
    if spec.event_archive and [dt.fromisoformat(r['issue_time']) for r in bundle.get('event_releases', [])] != expected:
        raise ValueError('Incomplete historical CME releases')
    if expected_products is not None and set(bundle['releases']) != set(expected_products):
        raise ValueError('Incomplete demo product set')
    for product, releases in bundle['releases'].items():
        if [dt.fromisoformat(r['issue_time']) for r in releases] != expected:
            raise ValueError(f'{product}: incomplete hourly demo releases')
    scenario = bundle.get('density_scenario')
    variants = {'jb2008_fixed', 'jb2008_forecast', 'dtm2020_fixed', 'dtm2020_forecast'}
    if not scenario or scenario.get('event') != bundle['event'] or set(scenario.get('releases', {})) != variants:
        raise ValueError('Missing or mismatched LEO scenario')
    for variant, releases in scenario['releases'].items():
        if [dt.fromisoformat(r['issue_time']) for r in releases] != expected:
            raise ValueError(f'{variant}: incomplete LEO issuances')
        for issue, release in zip(expected, releases):
            from datetime import timedelta
            times = [dt.fromisoformat(p['time']) for p in release['predictions']]
            if times != [issue + timedelta(hours=lead) for lead in range(1, 49)]:
                raise ValueError(f'{variant}: incomplete LEO horizon')


def rebuild(args, root=ROOT):
    inputs = preflight(args, root)
    destination = root / 'data/demo'
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / '.build.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another demo build is running; current dataset is unchanged') from None
        run = destination / 'runs' / datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
        pending = run / 'bundle/current.json'
        density = run / 'density'
        density.mkdir(parents=True)
        (density / 'gracefo').mkdir()
        for day in range(12, 25):
            stamp = f'202601{day:02d}'
            name = f'GF_OPER_DNS1ACC_2__{stamp}T000000_{stamp}T235959_0003.cdf'
            shutil.copy2(inputs['grace'] / name, density / 'gracefo' / name)
        print('[demo] 1/4: forecast products', flush=True)
        archives = forecast_stage(args, pending)
        raw_view = run / 'raw'
        (raw_view / 'omni_hourly').mkdir(parents=True)
        for path in archives:
            shutil.copy2(path, raw_view / 'omni_hourly' / path.name)
        (raw_view / 'goes').symlink_to(inputs['raw'] / 'goes', target_is_directory=True)
        print('[demo] 2/4: paired JB2008/DTM2020 baseline', flush=True)
        stage([root / 'scripts/training/thermosphere/run_event.py', '--manifest', inputs['manifest'],
               '--output', density, '--raw-root', raw_view, '--dtm-runtime', inputs['dtm']], root)
        print('[demo] 3/4: forecast geomagnetic drivers', flush=True)
        stage([root / 'scripts/training/thermosphere/run_event_forecast.py', '--experiment', density,
               '--demo-bundle', pending, '--dtm-runtime', inputs['dtm']], root)
        print('[demo] 4/4: warnings and verification', flush=True)
        stage([root / 'packages/intelligence-core/tools/density_event_replay.py',
               '--experiment', density / 'four-way'], root)
        bundle = json.loads(pending.read_text())
        bundle['density_scenario'] = json.loads((density / 'four-way/scenario.json').read_text())
        validate_complete(bundle, inputs.get('products'))
        from argus_prophet.demo import atomic_publish
        finished = datetime.now(UTC)
        bundle.update(version=finished.strftime('%Y%m%dT%H%M%S%fZ'), generated_at=finished.isoformat())
        atomic_publish(bundle, destination / 'current.json')
        print(f"[demo] Ready: /demo and /demo/leo · {bundle['version']} · evidence: {run}", flush=True)
        return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', nargs='?', default='january-2026')
    parser.add_argument('--refresh-data', action='store_true', help='Refresh OMNI archives; reuse pinned LEO caches')
    parser.add_argument('--raw-root', type=Path, help='Local OMNI and GOES cache root')
    parser.add_argument('--grace-root', type=Path, help='Local GRACE-C DNS ACC v0003 daily files')
    parser.add_argument('--dtm-runtime', type=Path, help='Licensed local research DTM2020 build')
    args = parser.parse_args()
    for name in ('raw_root', 'grace_root', 'dtm_runtime'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    # Keep scientific service boundaries separate; this host-side command orchestrates them.
    os.chdir(ROOT)
    os.environ.update(environment(ROOT))
    for path in os.environ['PYTHONPATH'].split(os.pathsep):
        if path not in sys.path: sys.path.insert(0, path)
    try:
        rebuild(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Demo rebuild failed; previous published dataset preserved. {error}\n')


if __name__ == '__main__': main()
