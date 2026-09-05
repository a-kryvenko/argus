import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from forecast.data_pipelines.solar_index_training import (
    load_calibration_dataset,
    training_observation_days,
)
from forecast.data_pipelines.solar_indices import (
    GOES_IRRADIANCE_COLUMNS,
    build_daily_goes_features,
    load_solar_index_calibrations,
)


NOTEBOOKS = Path(__file__).resolve().parents[3] / "notebooks"


def _run_notebook(name: str) -> dict:
    notebook = json.loads((NOTEBOOKS / name).read_text())
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{name}:cell-{index + 1}", "exec"), scope)
    return scope


def test_training_dates_are_deduplicated_utc_issue_dates(tmp_path):
    pd.DataFrame({
        "issue_time": ["2024-12-31T23:30:00-02:00"] * 2,
        "valid_time": ["2025-01-02T01:30Z", "2025-01-03T01:30Z"],
        "lead_hours": [24, 48],
    }).to_parquet(tmp_path / "part.parquet")
    assert training_observation_days(tmp_path).tolist() == [pd.Timestamp("2025-01-01T00:00Z")]


@pytest.fixture
def notebook_inputs(tmp_path, monkeypatch):
    import common.config
    import clio.dataloaders.spacewx_loader

    datasets = {}
    references = []
    rng = np.random.default_rng(42)
    for name, role, start in (("custom_train", "train", "2017-01-01"),
                              ("custom_holdout", "validation", "2025-01-01")):
        entry = {
            "role": role,
            "training_path": f"shards/{name}",
            "solfsmy_path": f"references/{name}.parquet",
            "goes_path": f"goes/{name}.parquet",
        }
        datasets[name] = entry
        dates = pd.date_range(start, periods=10, tz="UTC")
        training_path = tmp_path / entry["training_path"]
        training_path.mkdir(parents=True)
        # Duplicate lead rows must not add calibration samples or future days.
        pd.DataFrame({
            "issue_time": dates.repeat(2), "lead_hours": [1, 120] * len(dates),
            "valid_time": (dates + pd.Timedelta(days=5)).repeat(2),
        }).to_parquet(training_path / "part.parquet")
        goes = pd.DataFrame({
            "timestamp": dates + pd.Timedelta(hours=10),
            "goes_au_factor": 1.0, "goes_euvs_quality_valid": True,
            "goes_mgii_index": rng.uniform(0.25, 0.3, len(dates)),
            "goes_xray_background": rng.uniform(1e-7, 9e-7, len(dates)),
            **{col: rng.uniform(1, 10, len(dates)) for col in GOES_IRRADIANCE_COLUMNS},
        })
        goes_path = tmp_path / entry["goes_path"]
        goes_path.parent.mkdir(exist_ok=True)
        goes.to_parquet(goes_path)
        daily = build_daily_goes_features(goes)
        # Holdout has a deliberate +10 reference shift: fitting on it would hide bias.
        shift = 10 if role == "validation" else 0
        references.append(pd.DataFrame({
            "timestamp": daily.timestamp,
            "s10": 50 + 2 * daily.goes_euv_256_1au + shift,
            "m10": 20 + 400 * daily.goes_mgii_index + shift,
            "y10": 30 + daily.goes_lya_1216_1au + 1e7 * daily.goes_xray_background + shift,
        }))
    registry = {
        "model": "custom-calibration-v1",
        "calibration_path": "artifacts/custom-calibration.json",
        "metrics": "reports/custom-calibration",
        "datasets": datasets,
    }
    config = SimpleNamespace(workdir=tmp_path, models_registry={"models": {"solar_index_calibration": registry}})
    monkeypatch.setattr(common.config, "get_config", lambda: config)
    fetch = Mock(return_value=pd.concat(references, ignore_index=True))
    monkeypatch.setattr(clio.dataloaders.spacewx_loader, "fetch_spacewx", fetch)
    monkeypatch.setattr(plt, "show", lambda: None)
    yield config, registry, fetch
    plt.close("all")


def test_fetch_and_calibration_notebooks_follow_registry_and_preserve_holdout(notebook_inputs):
    config, registry, fetch = notebook_inputs
    fetching = _run_notebook("0_spacewx_data_fetching.ipynb")
    fetch.assert_called_once_with(
        start=pd.Timestamp("2017-01-01T00:00Z"), end=pd.Timestamp("2025-01-10T12:00Z"),
    )
    assert all(item["missing_reference_days"] == 0 for item in fetching["summary"])
    _run_notebook("0_spacewx_data_fetching.ipynb")
    assert fetch.call_count == 1  # Cache retained on a normal rerun.
    result = _run_notebook("4_calibrate_solar_indices.ipynb")
    np.testing.assert_allclose(result["metrics"]["bias"], -10, atol=1e-8)
    assert result["metrics"]["days"].tolist() == [10, 10, 10]
    assert load_solar_index_calibrations(config.workdir / registry["calibration_path"]) == result["calibrations"]
    reports = config.workdir / registry["metrics"]
    assert {path.name for path in reports.iterdir()} == {
        "validation.csv", "coverage.csv", "validation_predictions.parquet", "registry_snapshot.json",
    }
    for entry in registry["datasets"].values():
        original = pd.read_parquet(config.workdir / entry["training_path"])
        assert list(original.columns) == ["issue_time", "lead_hours", "valid_time"]
        assert len(original) == 20


def test_calibration_notebook_reports_missing_goes_before_export(notebook_inputs):
    config, registry, _ = notebook_inputs
    _run_notebook("0_spacewx_data_fetching.ipynb")
    registry["datasets"]["custom_train"]["goes_path"] = "missing-goes.parquet"
    with pytest.raises(FileNotFoundError, match="missing-goes.parquet"):
        _run_notebook("4_calibrate_solar_indices.ipynb")
    assert not (config.workdir / registry["calibration_path"]).exists()


def test_source_loader_restricts_goes_and_truth_to_shard_observation_days(notebook_inputs):
    config, registry, _ = notebook_inputs
    _run_notebook("0_spacewx_data_fetching.ipynb")
    entry = registry["datasets"]["custom_train"]
    path = config.workdir / entry["goes_path"]
    goes = pd.read_parquet(path)
    outside = goes.iloc[[0]].copy()
    outside["timestamp"] = pd.Timestamp("2025-01-01T10:00Z")
    pd.concat([goes, outside]).to_parquet(path)
    selected, truth = load_calibration_dataset(config.workdir, entry)
    assert len(selected) == len(truth) == 10
    assert selected.timestamp.dt.year.eq(2017).all()
