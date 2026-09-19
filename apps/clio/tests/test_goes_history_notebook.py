import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("forecast_core", reason="Private calibration checkout required")

if not (Path(__file__).resolve().parents[3] / "apps/clio/notebooks/0_goes_data_fetching.ipynb").exists():
    pytest.skip("Local ingestion notebooks required", allow_module_level=True)
import xarray as xr

from clio.dataloaders import goes_history_loader as loader
from forecast_core.data_pipelines.solar_indices import build_daily_goes_features


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
    daily = build_daily_goes_features(joined)
    assert daily.goes_euv_256_1au.iloc[0] == 2
    assert daily.goes_valid_sample_count.iloc[0] == 1














def test_history_notebook_writes_registry_splits_and_reports_missing_days(tmp_path, monkeypatch):
    import common.config

    euvs = _euvs_file(tmp_path / "euvs.nc")
    xrs = _xrs_file(tmp_path / "xrs.nc")
    sources = [loader.GoesArchiveFile(16, loader.EUVS_PRODUCT, "https://example.test/euvs.nc", 2025),
               loader.GoesArchiveFile(16, loader.XRS_PRODUCT, "https://example.test/xrs.nc")]
    discover = Mock(return_value=sources)
    monkeypatch.setattr(loader, "discover_goes_history", discover)
    monkeypatch.setattr(loader, "cache_archive_file", lambda source, *args, **kwargs:
                        euvs if source.product == loader.EUVS_PRODUCT else xrs)
    training = tmp_path / "shards"
    training.mkdir()
    pd.DataFrame({"issue_time": pd.to_datetime(["2010-01-01T00:00Z", "2025-01-01T00:00Z", "2025-01-01T00:00Z"])}).to_parquet(training / "part.parquet")
    registry = {"datasets": {"test": {"training_path": "shards", "goes_path": "result/goes.parquet"}},
                "goes_archive": {"cache_path": "cache", "satellites": [16], "url": "https://example.test/"},
                "metrics": "reports"}
    monkeypatch.setattr(common.config, "get_config", lambda: SimpleNamespace(workdir=tmp_path, models_registry={"models": {"solar_index_calibration": registry}}))
    notebook_path = Path(__file__).resolve().parents[3] / "apps/clio/notebooks/0_goes_data_fetching.ipynb"
    notebook = json.loads(notebook_path.read_text())

    def run():
        scope = {}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"notebook-cell-{index}", "exec"), scope)
        return scope

    result = run()
    saved = pd.read_parquet(tmp_path / "result/goes.parquet")
    assert len(saved) == 4
    assert result["summary"][0]["days_without_valid_samples"] == 1
    assert (tmp_path / "reports/goes_archive_manifest.csv").is_file()
    assert (tmp_path / "reports/goes_daily_coverage.csv").is_file()
    assert len(build_daily_goes_features(saved)) == 1
    run()  # Completed normalized output skips archive discovery/download.
    assert discover.call_count == 1
