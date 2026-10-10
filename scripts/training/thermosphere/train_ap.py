"""Freeze a causal experimental ap median model before the density test period.

The legacy model lacks embedded train dates. This explicitly dated CPU refit uses
the same 26-feature family, a single lead-conditioned LightGBM and fixed settings.
There is no tuning on 2025 ap or density. No production artifacts are replaced.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from prepare import HERE, ROOT, digest

CONFIG = json.loads((HERE / "config_three_way.json").read_text())
OUT = ROOT / "data/experiments" / CONFIG["experiment"]
FEATURES = ["lead_hours", "southward_bz", "v_mean_6h", "v_delta_6h",
    "bt", "bt_mean_3h", "bt_delta_3h", "bt_mean_6h", "bt_delta_6h", "bt_mean_7d", "bt_delta_7d",
    "ap", "ap_mean_3h", "ap_delta_3h", "ap_mean_6h", "ap_delta_6h", "ap_mean_7d", "ap_delta_7d",
    "f10_7", "dst", "dst_mean_3h", "dst_delta_3h", "dst_mean_6h", "dst_delta_6h", "dst_mean_7d", "dst_delta_7d"]


def read_omni():
    paths = [ROOT/f"data/raw/omni_{split}.csv" for split in ("2010_2024", "2025")]
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw["issue_time"] = pd.to_datetime(raw.issue_time, utc=True)
    raw = raw.set_index("issue_time").sort_index()
    if raw.index.has_duplicates:
        raise ValueError("Duplicate OMNI timestamps")
    raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="h"))
    # Archives normally contain NaN sentinels; retain gaps instead of smoothing.
    for col, low, high in [("ap", 0, 400), ("kp", 0, 9), ("dst", -2000, 100),
                           ("f10_7", 30, 500), ("v", 100, 3000),
                           ("bx", -500, 500), ("by", -500, 500), ("bz", -500, 500)]:
        raw[col] = raw[col].where(raw[col].between(low, high))
    return raw, paths


def causal_features(raw):
    """All rows are issue-time features from closed archive intervals.

    Hourly wind/IMF/Dst: shift one hour. ap/Kp: only the last closed UTC 3h
    bin, not an in-progress repeated OMNI value. F107: last closed UTC day.
    Rolling means run over this availability-aligned hourly history.
    """
    from forecast.inputs.observations import build_features
    clock = raw.index
    if not clock.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Require a contiguous hourly archive grid")
    available = raw.shift(1).copy()
    for col in ("ap", "kp"):
        bins = raw[col].resample("3h").first()
        if (raw[col].resample("3h").nunique() > 1).any():
            raise ValueError(f"Inconsistent {col} bins")
        bins.index += pd.Timedelta(hours=3)
        available[col] = bins.reindex(clock, method="ffill", tolerance=pd.Timedelta(hours=3))
    days = raw.f10_7.resample("D").first()
    if (raw.f10_7.resample("D").nunique() > 1).any():
        raise ValueError("Inconsistent daily F107")
    days.index += pd.Timedelta(days=1)
    available["f10_7"] = days.reindex(clock, method="ffill", tolerance=pd.Timedelta(days=1))
    return build_features(available)[FEATURES[1:]]


def training_rows(raw, features):
    cutoff = pd.Timestamp(CONFIG["train_end_exclusive"])
    start = pd.Timestamp(CONFIG["train_start"])
    issues = features.loc[(features.index >= start) & (features.index < cutoff)].iloc[
        ::CONFIG["training_issue_stride_hours"]].dropna().index
    frames = []
    for lead in range(1, CONFIG["max_lead_hours"]+1):
        valid = issues+pd.Timedelta(hours=lead)
        # ap targets denote 3-hour interval starts; the entire target bin must
        # have closed before the training cutoff, not just its label timestamp.
        released = valid.floor("3h")+pd.Timedelta(hours=3)
        targets = raw.ap.reindex(valid).to_numpy()
        mask = (released < cutoff) & np.isfinite(targets)
        rows = features.loc[issues[mask]].copy()
        rows["lead_hours"] = lead
        rows["target_ap"] = targets[mask]
        rows["valid_time"] = valid[mask]
        rows["label_available_at"] = released[mask]
        rows.index.name = "issue_time"
        frames.append(rows.reset_index())
    return pd.concat(frames, ignore_index=True)


def fit_model(raw, features, paths):
    began = time.monotonic()
    rows = training_rows(raw, features)
    print(f"Fit experimental ap q50: {len(rows):,} rows, latest label release {rows.label_available_at.max()}", flush=True)
    model = LGBMRegressor(**CONFIG["model_params"])
    model.fit(rows[FEATURES].astype(np.float32), rows.target_ap.to_numpy(dtype=np.float32))
    metadata = {
        "config": CONFIG, "feature_columns": FEATURES,
        "training_rows": len(rows), "training_issues": rows.issue_time.nunique(),
        "min_issue": rows.issue_time.min().isoformat(), "max_issue": rows.issue_time.max().isoformat(),
        "max_label_time": rows.valid_time.max().isoformat(),
        "max_label_available_at": rows.label_available_at.max().isoformat(),
        "source_files": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "training_code_sha256": digest(HERE/"train_ap.py"),
        "feature_code_sha256": digest(ROOT/"packages/forecast/src/forecast/inputs/observations.py"),
        "training_matrix_sha256": __import__("hashlib").sha256(
            pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes()).hexdigest(),
        "elapsed_seconds": time.monotonic()-began,
        "note": "New experimental median model, not the undated production ap artifact. No test-set tuning.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "metadata": metadata}
    joblib.dump(bundle, OUT/"ap_q50.joblib")
    metadata["artifact_sha256"] = digest(OUT/"ap_q50.joblib")
    (OUT/"ap_training.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"AP training finished in {metadata['elapsed_seconds']:.1f}s", flush=True)
    return bundle


def forecast_ap(bundle, features, issues):
    issues = pd.DatetimeIndex(sorted(set(issues)))
    if pd.Timestamp(bundle["metadata"]["max_label_available_at"]) >= issues.min():
        raise ValueError("Training labels overlap forecast issuance")
    # Half-hour issues use only the last completed hourly feature snapshot.
    base = features.reindex(issues.floor("h"))
    base.index = issues
    missing = base.isna().any(axis=1)
    frames = []
    for lead in range(1, CONFIG["max_lead_hours"]+1):
        frame = base.loc[~missing].copy()
        frame["lead_hours"] = lead
        prediction = bundle["model"].predict(frame[FEATURES].astype(np.float32))
        frames.append(pd.DataFrame({"issue_time": frame.index,
            "valid_time": frame.index+pd.Timedelta(hours=lead), "lead_hours": lead,
            "ap_q50": prediction}))
    return pd.concat(frames, ignore_index=True), pd.DataFrame({"issue_time": issues[missing], "reason": "missing_ap_model_features"})


if __name__ == "__main__":
    raw, paths = read_omni()
    features = causal_features(raw)
    fit_model(raw, features, paths)
