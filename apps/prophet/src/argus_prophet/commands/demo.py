"""One-command demo preparation, independent of operational databases and writers."""
import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, UTC
from pathlib import Path


def archive(year, config, folder, refresh=False, *, source_root=None):
    from common.data.omni import parse_annual
    destination = folder / f'omni2_{year}.dat'
    candidates = [destination, config.data_root / 'raw/omni_hourly' / destination.name,
                  config.data_root / 'mnt/raw/omni_hourly' / destination.name,
                  config.data_root / 'demo/january-2026' / destination.name]
    if source_root is not None:
        candidates = [Path(source_root) / 'omni_hourly' / destination.name]
    if not refresh:
        for candidate in candidates:
            if candidate.is_file():
                parse_annual(candidate.read_text(encoding='ascii'), year)
                print(f'OMNI {year}: using cached archive', flush=True)
                return candidate
    import httpx
    url = f'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat'
    print(f'OMNI {year}: downloading {url}', flush=True)
    response = httpx.get(url, timeout=120, follow_redirects=True)
    response.raise_for_status()
    parse_annual(response.content.decode('ascii'), year)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(response.content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(args, *, output=None, archive_paths=None, raw_root=None):
    # Set these before loading numerical runtimes, including PyTorch/LightGBM.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    from common.config import get_config
    from argus_prophet.demo import Manifest, build, validate_evidence
    from argus_prophet.demo_sources import prepare

    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', args.scenario):
        raise ValueError('Scenario must be a name such as january-2026')
    config = get_config()
    path = config.config_root / 'demo' / f'{args.scenario}.json'
    if not path.is_file():
        raise ValueError(f'Demo scenario not found: {args.scenario}')
    manifest = Manifest.model_validate_json(path.read_text())
    validate_evidence(manifest)
    # Fail before network traffic if the model/evidence configuration is stale.
    for name, evidence in manifest.models.items():
        entry = config.models_registry['models'][name]
        model = config.data_root / 'models' / (entry['model'] + '.joblib')
        if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest() != evidence.sha256:
            raise ValueError(f'{name}: model missing or changed; update its training evidence before rebuilding demo')
    root = config.data_root / 'demo'
    root.mkdir(parents=True, exist_ok=True)
    output = root / 'current.json' if output is None else Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / '.build.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another demo build is running; current demo remains available') from None
        try:
            years = [manifest.event.starts_at.year - 1, manifest.event.starts_at.year]
            archive_options = {} if raw_root is None else {'source_root': raw_root}
            archives = [archive(year, config, root / 'archives', args.refresh_data, **archive_options) for year in years]
            if archive_paths is not None:
                archive_paths.extend(archives)
            inputs = root / args.scenario / 'runs' / datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
            prepare(path, archives, inputs)
            bundle = build(manifest, json.loads((inputs / 'observations.json').read_text()),
                           output, inputs / 'snapshots')
            status = 'Demo ready' if output == root / 'current.json' else 'Demo forecasts staged'
            print(f"{status}: {len(bundle['releases'])} products · version {bundle['version']}", flush=True)
        except Exception as exc:
            raise ValueError(f'Demo build failed; previous published dataset preserved. {exc}') from exc
