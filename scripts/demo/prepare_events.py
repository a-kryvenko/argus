"""Freeze publication-gated single-CME DONKI notifications for the demo."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare(destination):
    source = ROOT / 'data/experiments/proswin_cme_v1/raw'
    index = source / 'list-2026-01.json'
    entries = json.loads(index.read_text())
    files = [source / ('notification-' + r['id'] + '.json') for r in entries
        if r['sendDate'] < '2026-01-23' and any(e['type'] == 'ENLIL' for e in r.get('linkedEvents', []))]
    parser = ROOT / 'notebooks/workflows/solar_events/enlil_context_features.py'
    spec = importlib.util.spec_from_file_location('enlil_notifications', parser)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records, audit = module.notifications(files)
    rows = []
    for r in records:
        if r['completion'] is None: continue
        rows.append(dict(event_id=r['event_id'], available_at=r['available'].isoformat(),
            observed_at=r['start'].isoformat(), model_completed_at=r['completion'].isoformat(),
            earth_arrival_at=r['arrival'].isoformat() if r['arrival'] is not None else None,
            duration_hours=r['duration'], glancing_blow=r['glancing'], minor_impact=r['minor'],
            initial_speed_km_s=float(r['speed']) if module.np.isfinite(r['speed']) else None,
            simulation_id=r['event_id'] + '@' + ':'.join(str(v) for v in r['run_order']),
            source_url='https://kauai.ccmc.gsfc.nasa.gov/DONKI/'))
    archive = dict(schema_version=1, availability_basis='provider_notification_publication_not_recorded_receipt',
        coverage_start='2026-01-15T19:00:00Z', coverage_end='2026-01-22T19:00:00Z',
        scope='single_CME_notifications', audit=audit, records=rows,
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [index, parser, *files]])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(archive, indent=2, allow_nan=False)+'\n')
    return hashlib.sha256(destination.read_bytes()).hexdigest()


if __name__ == '__main__':
    print(prepare(ROOT / 'data/demo/january-2026/enlil-notifications.json'))
