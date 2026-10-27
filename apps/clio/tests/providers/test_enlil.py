import io

import numpy as np
import pytest
from scipy.io import netcdf_file

from clio.providers import enlil


def profile(*, label='l1', unit='m/s', times=(0., 3600.), speed=(400000., 800000.)):
    buffer = io.BytesIO()
    with netcdf_file(buffer, 'w') as f:
        f.label, f.type = label, 'EVO'
        f.rundate_cal, f.rundate_mjd = '2026-01-01 00:00:00', 61041.
        f.creation, f.version = '2026-01-01 00:10:00', '2.9'
        f.createDimension('time', 2)
        for name, values, units in [('TIME', times, 's'), ('D', (5*enlil.PROTON_MASS_KG*1e6,)*2, 'kg/m3'),
                                     ('V1', speed, unit), ('V2', (0., 0.), 'm/s'), ('V3', (0., 0.), 'm/s')]:
            v = f.createVariable(name, 'd', ('time',))
            v.units, v[:] = units, values
        f.flush()
        return buffer.getvalue()


def test_classic_netcdf_units_density_and_timestamps():
    result = enlil.parse_profile(profile())
    assert result['target'] == 'L1' and result['density_kind'] == 'proton_equivalent'
    assert [p['speed_km_s'] for p in result['points']] == [400, 800]
    assert [p['density_cm3'] for p in result['points']] == pytest.approx([5, 5])
    assert result['points'][1]['valid_time'] == '2026-01-01T01:00:00+00:00'


@pytest.mark.parametrize('kwargs', [dict(label='earth'), dict(unit='km/s'),
    dict(times=(3600., 0.)), dict(speed=(np.nan, 400000.)), dict(speed=(np.inf, 400000.))])
def test_wrong_location_units_and_invalid_coordinates_are_rejected(kwargs):
    with pytest.raises(ValueError):
        enlil.parse_profile(profile(**kwargs))


def test_discovery_never_fetches_full3d_and_prefers_l1(monkeypatch):
    from datetime import UTC, datetime
    calls = []
    run = 'wsa_enlil.20261014_1/'
    model = run+'wsa_enlil_1.test/'
    def listing(prefix, delimiter=None):
        calls.append(prefix)
        if prefix == 'wsa_enlil.20261014':
            return [run], []
        if prefix == run:
            return [model], []
        if prefix == model+'evo.l1.nc':
            return [], [dict(key=prefix, size=100, last_modified='2026-10-14T00:00Z')]
        return [], []
    monkeypatch.setattr(enlil, 'listing', listing)
    result = enlil.discover(datetime(2026, 10, 14, tzinfo=UTC))
    assert result[0]['key'] == model+'evo.l1.nc'
    assert not any('tar' in p or p.endswith('evo.earth.nc') for p in calls)


def test_collection_preserves_raw_rejections_and_success_independently(monkeypatch, tmp_path):
    monkeypatch.setenv("ARGUS_ENLIL_ARCHIVE", str(tmp_path))
    import asyncio
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    from clio.ingestion import enlil as collector
    good, bad = profile(), b'invalid NetCDF'
    objects = [dict(key=key, size=len(raw), last_modified='2026-01-01T00:15Z')
               for key, raw in [('good', good), ('bad', bad)]]
    monkeypatch.setattr(enlil, 'discover', lambda now: objects)
    monkeypatch.setattr(enlil, 'download', lambda url: good if url.endswith('good') else bad)
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.scalar.return_value = None
    monkeypatch.setattr(collector, 'get_db_session', lambda: session)
    saved = []
    async def store(values):
        saved.append(values)
    monkeypatch.setattr(collector, 'store', store)
    with pytest.raises(RuntimeError, match='1 profiles'):
        asyncio.run(collector.collect())
    from clio.observations.archive import read_receipt, read_original
    row = SimpleNamespace(**saved[0])
    assert read_receipt(row)['payload']['points'][0]['speed_km_s'] == 400
    assert read_original(row.kind, row.raw_path, row.sha256) == good
    assert saved[1]['error'] and saved[1]['raw_path'] and saved[1]['sha256']
