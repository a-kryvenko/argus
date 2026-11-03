import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from clio.providers import goes_history as loader


def _euvs_file(path: Path, satellite=16):
    time = pd.date_range("2025-01-01", periods=4, freq="min")
    variables = {}
    for line in ("256", "284", "304", "1175", "1216", "1335", "1405"):
        variables[f"irr_{line}"] = ("time", [1.0, 2.0, 3.0, 4.0])
        variables[f"irr_{line}_flag"] = ("time", [0, 0, 0, 0])
    variables["irr_284_flag"] = ("time", [0, 1, 0, 0])
    variables.update({
        "MgII_standard": ("time", [0.28] * 4),
        "MgII_EXIS": ("time", [0.37] * 4),
        "MgII_flag": ("time", [0, 0, 0, 4]),
        "geocorona_flag": ("time", [0, 0, 1, 0]),
        "au_factor": ("time", [2.0] * 4),
    })
    xr.Dataset(variables, coords={"time": time}, attrs={"platform": f"g{satellite}"}).to_netcdf(path, engine="h5netcdf")
    return path


def _xrs_file(path: Path, satellite=16):
    xr.Dataset({
        "bkd1d_xrsb_flux": ("time", [1e-7, 9e-7, 3e-7]),
        "avg1d_xrsb_flux": ("time", [1e-5] * 3),
        "bkd1d_xrsb_flag": ("time", [0, 0, 1]),
    }, coords={"time": pd.date_range("2024-12-31", periods=3)},
        attrs={"platform": f"g{satellite}"}).to_netcdf(path, engine="h5netcdf")
    return path


def test_normalization_uses_standard_mgii_flags_and_previous_day_background(tmp_path):
    euvs = loader.read_euvs_archive(_euvs_file(tmp_path / "euvs.nc"), 16)
    xrs = loader.read_xrs_archive(_xrs_file(tmp_path / "xrs.nc"), 16)
    assert euvs.goes_mgii_index.dropna().tolist() == [0.28, 0.28]
    assert euvs.goes_euv_284.isna().tolist() == [False, True, False]
    assert euvs.timestamp.tolist() == [pd.Timestamp('2025-01-01T00:00Z'),
        pd.Timestamp('2025-01-01T00:01Z'), pd.Timestamp('2025-01-01T00:03Z')]
    assert not any('quality' in name or 'flag' in name for name in euvs)
    assert len(xrs) == 2
    joined = loader.join_archive_background(euvs, xrs)
    assert joined.goes_xray_background.eq(1e-7).all()  # Neither today's final value nor daily average.
    assert joined.goes_xray_background_timestamp.eq(pd.Timestamp("2024-12-31T00:00Z")).all()
    assert joined.goes_euv_256.iloc[0] == 1  # 1-AU correction deferred to the shared builder.


def test_missing_background_is_not_forward_filled(tmp_path):
    euvs = loader.read_euvs_archive(_euvs_file(tmp_path / "euvs.nc"), 16)
    xrs = loader.read_xrs_archive(_xrs_file(tmp_path / "xrs.nc"), 16)
    euvs.timestamp += pd.Timedelta(days=2)
    assert loader.join_archive_background(euvs, xrs).goes_xray_background.isna().all()


def test_satellite_selection_uses_validity_before_priority():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01T00:00Z"] * 2 + ["2025-01-01T00:01Z"] * 2),
        "goes_euvs_satellite": [18, 16, 18, 16],
        "goes_xray_background": [float("nan"), 1e-7, 1e-7, 1e-7],
    })
    chosen = loader.select_satellite_records(frame, [18, 16])
    assert chosen.goes_euvs_satellite.tolist() == [16, 18]
    assert not chosen.timestamp.duplicated().any()


def test_discovery_chooses_numeric_versions_and_does_not_fabricate_missing_years():
    listings = [
        '<a href="sci_euvs-l2-avg1m_g16_y2025_v1-0-9.nc">old</a>'
        '<a href="sci_euvs-l2-avg1m_g16_y2025_v1-0-10.nc">new</a>'
        '<a href="sci_euvs-l2-avg1m_g16_y2024_v1-0-9.nc">unrequested</a>'
        '<a href="https://unrelated.test/sci_euvs-l2-avg1m_g16_y2025_v1-0-99.nc">external</a>',
        '<a href="sci_xrsf-l2-bkd1d_g16_s20170207_e20250406_v2-2-1.nc">background</a>',
    ]
    session = Mock()
    session.get.side_effect = [Mock(status_code=200, text=text) for text in listings]
    files = loader.discover_goes_history([2010, 2025], [16], session)
    assert len(files) == 2
    assert files[0].year == 2025
    assert files[0].url.endswith("v1-0-10.nc")


def test_cache_reuses_complete_downloads(tmp_path):
    payload = _euvs_file(tmp_path / "source.nc").read_bytes()
    response = Mock()
    response.iter_content.return_value = [payload]
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    session = Mock()
    session.get.return_value = context
    source = loader.GoesArchiveFile(16, loader.EUVS_PRODUCT, "https://example.test/source.nc", 2025)
    cached = loader.cache_archive_file(source, tmp_path / "cache", session)
    assert cached.read_bytes() == payload
    assert loader.cache_archive_file(source, tmp_path / "cache", session) == cached
    assert session.get.call_count == 1


def test_interrupted_download_does_not_replace_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    original = _euvs_file(cache / "source.nc").read_bytes()
    response = Mock()
    response.iter_content.return_value = [b"not a netcdf file"]
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    session = Mock()
    session.get.return_value = context
    source = loader.GoesArchiveFile(16, loader.EUVS_PRODUCT, "https://example.test/source.nc", 2025)
    with pytest.raises((OSError, ValueError)):
        loader.cache_archive_file(source, cache, session, refresh=True)
    assert (cache / "source.nc").read_bytes() == original
    assert list(cache.iterdir()) == [cache / "source.nc"]


def test_refresh_checks_unchanged_source_without_downloading_body(tmp_path):
    path = _euvs_file(tmp_path / 'source.nc')
    before = path.read_bytes()
    path.with_suffix('.nc.http.json').write_text(json.dumps({'etag': 'original-version',
        'last_modified': 'Thu, 01 Jan 2026 00:00:00 GMT'}))
    response = Mock(status_code=304)
    context = Mock(__enter__=Mock(return_value=response), __exit__=Mock(return_value=False))
    session = Mock(get=Mock(return_value=context))
    source = loader.GoesArchiveFile(16, loader.EUVS_PRODUCT, 'https://example.test/source.nc', 2025)
    assert loader.cache_archive_file(source, tmp_path, session, refresh=True) == path
    assert path.read_bytes() == before
    response.iter_content.assert_not_called()
    assert session.get.call_args.kwargs['headers'] == {
        'If-None-Match': 'original-version', 'If-Modified-Since': 'Thu, 01 Jan 2026 00:00:00 GMT'}


def test_cached_archives_covering_missing_snapshot_days_need_no_http_refresh(tmp_path):
    _euvs_file(tmp_path / 'euvs.nc')
    _xrs_file(tmp_path / 'xrs.nc')
    files = [loader.GoesArchiveFile(16, loader.EUVS_PRODUCT, 'https://example.test/euvs.nc', 2025),
             loader.GoesArchiveFile(16, loader.XRS_PRODUCT, 'https://example.test/xrs.nc')]
    session = Mock()
    frames = list(loader.iter_goes_history(files, pd.date_range('2025-01-01', periods=1, tz='UTC'),
                                          [16], tmp_path, session, refresh=True))
    assert len(frames) == 1 and len(frames[0]) == 3
    session.get.assert_not_called()


def test_cached_annual_file_is_revalidated_only_when_requested_day_is_absent(monkeypatch, tmp_path):
    _euvs_file(tmp_path / 'euvs.nc')
    _xrs_file(tmp_path / 'xrs.nc')
    files = [loader.GoesArchiveFile(16, loader.EUVS_PRODUCT, 'https://example.test/euvs.nc', 2025),
             loader.GoesArchiveFile(16, loader.XRS_PRODUCT, 'https://example.test/xrs.nc')]
    cache = Mock(side_effect=lambda source, *a, **kw: tmp_path / Path(source.url).name)
    monkeypatch.setattr(loader, 'cache_archive_file', cache)
    list(loader.iter_goes_history(files, pd.date_range('2025-01-02', periods=1, tz='UTC'),
                                 [16], tmp_path, Mock(), refresh=True))
    refreshed = [call.args[0].product for call in cache.call_args_list if call.kwargs.get('refresh')]
    assert refreshed == [loader.EUVS_PRODUCT]
