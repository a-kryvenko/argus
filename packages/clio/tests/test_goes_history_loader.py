import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from clio.dataloaders import goes_history_loader as loader


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
    assert euvs.goes_mgii_index.tolist() == [0.28] * 4
    assert euvs.goes_euvs_quality_valid.tolist() == [True, False, False, False]
    assert euvs.goes_eclipse.tolist() == [False, True, False, False]
    assert euvs.goes_geocorona.tolist() == [False, False, True, False]
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
        "goes_euvs_quality_valid": [False, True, True, True],
        "goes_xray_background": [1e-7] * 4,
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
