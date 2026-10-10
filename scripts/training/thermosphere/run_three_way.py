"""Run the three-variant held-out density comparison without changing production.

.venv/bin/python -u scripts/training/thermosphere/train_ap.py
.venv/bin/python -u scripts/training/thermosphere/run_three_way.py

The first two variants are checksum-verified outputs of run.py; the third changes
only geomagnetic inputs using a pre-test frozen ap median forecast. F107 persists.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from ap_forecast import forecast_dtm_snapshot
from models import DTM2020
from prepare import HERE, ROOT, digest
from train_ap import CONFIG, OUT, causal_features, forecast_ap, read_omni

MODELS = ("jb2008", "dtm2020", "dtm2020_ap_forecast")
BASE = ROOT/"data/experiments"/CONFIG["baseline_experiment"]


def common_rows(predictions):
    return predictions.loc[np.isfinite(predictions[list(MODELS)]).all(axis=1)
        & predictions[list(MODELS)].gt(0).all(axis=1) & predictions.truth.gt(0)].copy()


def metrics(frame):
    from run import paired_metrics
    output = []
    for grouping in [("lead_hours",), ("lead_hours", "activity"), ("lead_hours", "window")]:
        for key, group in frame.groupby(list(grouping)):
            key = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(grouping, key))
            base = paired_metrics(group)
            # Reuse exactly the established metric formulas for the third variant.
            third = paired_metrics(group.drop(columns="dtm2020").rename(columns={"dtm2020_ap_forecast": "dtm2020"}))[1]
            third["model"] = "dtm2020_ap_forecast"
            output.extend({"grouping": "+".join(grouping), **labels, **row} for row in [*base, third])
    return pd.DataFrame(output)


def differences(frame):
    rng = np.random.default_rng(CONFIG["seed"])
    output = []
    for lead, group in frame.groupby("lead_hours"):
        for model, reference in [("dtm2020", "jb2008"), ("dtm2020_ap_forecast", "dtm2020"),
                                 ("dtm2020_ap_forecast", "jb2008")]:
            delta = 100*((group[model]/group.truth-1).abs()-(group[reference]/group.truth-1).abs())
            daily = delta.groupby(group.valid_time.dt.floor("D")).agg(["sum", "count"])
            samples = rng.integers(0, len(daily), (2000, len(daily)))
            boot = daily["sum"].to_numpy()[samples].sum(axis=1)/daily["count"].to_numpy()[samples].sum(axis=1)
            output.append({"lead_hours": int(lead), "model": model, "reference": reference,
                "mape_difference_pp": float(delta.mean()),
                "ci95_low": float(np.quantile(boot, .025)), "ci95_high": float(np.quantile(boot, .975))})
    return pd.DataFrame(output)


def check_baseline():
    baseline = json.loads((BASE/"results.json").read_text())
    for item in baseline["inputs"]:
        if digest(ROOT/item["path"]) != item["sha256"]:
            raise ValueError(f"Changed baseline input: {item['path']}")
    for name, sha in baseline["production_source_hashes"].items():
        if digest(ROOT/"packages/forecast-core/src/forecast_core"/name) != sha:
            raise ValueError(f"Production baseline source changed: {name}")
    build = json.loads((BASE/"build.json").read_text())
    if digest(BASE/"libdtm2020.so") != build["library_sha256"]:
        raise ValueError("DTM library has changed")
    for artifact in json.loads((BASE/"download_manifest.json").read_text()):
        if digest(ROOT/artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Changed downloaded source: {artifact['path']}")
    return baseline


def main():
    began = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = check_baseline()
    training = json.loads((OUT/"ap_training.json").read_text())
    if digest(OUT/"ap_q50.joblib") != training["artifact_sha256"]:
        raise ValueError("AP model changed after training")
    if training["config"] != CONFIG or training["training_code_sha256"] != digest(HERE/"train_ap.py"):
        raise ValueError("AP training configuration/code changed; rerun train_ap.py")
    bundle = joblib.load(OUT/"ap_q50.joblib")
    predictions = pd.read_parquet(BASE/"predictions.parquet")
    observations = pd.read_parquet(BASE/"observations.parquet")
    measurements = pd.read_parquet(BASE/"measurements.parquet")
    baseline_drivers = {pd.Timestamp(row["issue_time"]): row["dtm2020"]
                        for row in json.loads((BASE/"driver_snapshots.json").read_text())}
    raw, paths = read_omni()
    for path in paths:
        if digest(path) != training["source_files"][str(path.relative_to(ROOT))]:
            raise ValueError("OMNI changed since AP training")
    features = causal_features(raw)
    ap, missing = forecast_ap(bundle, features, predictions.issue_time.unique())
    ap.to_parquet(OUT/"ap_forecasts.parquet", index=False)
    missing.to_csv(OUT/"ap_forecast_missing.csv", index=False)
    features.reindex(pd.DatetimeIndex(predictions.issue_time.unique()).floor("h")).to_parquet(OUT/"ap_input_features.parquet")
    groups = {issue: frame for issue, frame in ap.groupby("issue_time")}
    predictions = predictions.merge(observations[["valid_time", "local_solar_time"]], on="valid_time", validate="many_to_one")
    dtm = DTM2020(BASE)
    values, reasons, snapshots = [], [], []
    known = {}
    print(f"Evaluate {len(predictions)} density rows; {len(groups)} forecast issues; {len(missing)} unavailable", flush=True)
    for i, row in enumerate(predictions.itertuples(index=False)):
        d = baseline_drivers[row.issue_time]
        if row.lead_hours == 0:
            values.append(row.dtm2020)
            reasons.append("")
            snapshots.append({"issue_time": row.issue_time, "valid_time": row.valid_time, "lead_hours": row.lead_hours, **d})
            continue
        if row.issue_time not in groups:
            values.append(np.nan)
            reasons.append("missing_ap_model_features")
            continue
        if row.issue_time not in known:
            known[row.issue_time] = measurements.loc[measurements.metric.eq("ap")
                & measurements.available_at.le(row.issue_time)
                & measurements.observed_at.ge(row.issue_time-pd.Timedelta(hours=30))]
        try:
            drivers = forecast_dtm_snapshot(known[row.issue_time], groups[row.issue_time],
                row.issue_time, row.valid_time, baseline=d)
            day = row.valid_time.dayofyear+(row.valid_time-row.valid_time.floor("D")).total_seconds()/86400
            density = dtm.evaluate(day, row.altitude_km, np.deg2rad(row.latitude_deg),
                np.deg2rad(row.longitude_deg), row.local_solar_time*np.pi/12,
                drivers["f107"], drivers["f107_mean"], drivers["kp"], drivers["kp_mean"])[0]
            values.append(float(density))
            reasons.append("")
            snapshots.append({"issue_time": row.issue_time, "valid_time": row.valid_time, "lead_hours": row.lead_hours, **drivers})
        except ValueError as error:
            values.append(np.nan)
            reasons.append(str(error))
        if i % 1000 == 0:
            print(f"{i}/{len(predictions)} rows, elapsed {time.monotonic()-began:.0f}s", flush=True)
    predictions["dtm2020_ap_forecast"] = values
    predictions["dtm2020_ap_forecast_error"] = reasons
    predictions.to_parquet(OUT/"predictions.parquet", index=False)
    pd.DataFrame(snapshots).to_parquet(OUT/"driver_snapshots.parquet", index=False)
    matched = common_rows(predictions)
    if matched.empty:
        raise ValueError("No common observations across three variants")
    scores = metrics(matched)
    scores.to_csv(OUT/"metrics.csv", index=False)
    differences(matched).to_csv(OUT/"paired_differences.csv", index=False)
    coverage = predictions.groupby("lead_hours")[list(MODELS)].count()
    coverage["requested"] = predictions.groupby("lead_hours").size()
    coverage["common"] = matched.groupby("lead_hours").size()
    coverage.to_csv(OUT/"coverage.csv")
    if coverage["common"].isna().any() or coverage["common"].le(0).any():
        raise ValueError("At least one forecast horizon has no common predictions; inspect coverage/errors")
    # Evaluate the driver separately; this diagnosis does not feed density forecasts.
    ap["truth"] = raw.ap.reindex(pd.DatetimeIndex(ap.valid_time).floor("h")).to_numpy()
    ap["persistence"] = raw.ap.reindex(pd.DatetimeIndex(ap.issue_time).floor("3h")-pd.Timedelta(hours=3)).to_numpy()
    valid = ap.dropna(subset=["truth", "persistence"]).copy()
    valid["forecast_abs_error"] = (valid.ap_q50.clip(0, 400)-valid.truth).abs()
    valid["persistence_abs_error"] = (valid.persistence-valid.truth).abs()
    valid.groupby("lead_hours")[["forecast_abs_error", "persistence_abs_error"]].mean().to_csv(OUT/"ap_metrics.csv")
    provenance = {
        "config": CONFIG, "baseline_config": baseline["config"], "elapsed_seconds": time.monotonic()-began,
        "training": training, "models": list(MODELS),
        "baseline_artifact_hashes": {name: digest(BASE/name) for name in ["predictions.parquet", "results.json",
            "observations.parquet", "measurements.parquet", "driver_snapshots.json", "libdtm2020.so"]},
        "scripts": {p.name: digest(p) for p in HERE.glob("*.py")},
        "outputs": {name: digest(OUT/name) for name in ["predictions.parquet", "metrics.csv", "ap_forecasts.parquet"]},
        "ap_clipped_count": int((~ap.ap_q50.between(0, 400)).sum()),
        "limitations": [
            "Same GRACE-FO 1 windows and measured future positions as baseline; not an orbit/drag-risk validation.",
            "2025 density and ap targets were not used for model fitting or hyperparameter selection.",
            "Experimental dated LightGBM ap q50 refit, not the undated production ap model.",
            "Variant 3 forecasts only ap-derived Kp; F107 and its background remain fixed, identical to variant 2.",
            "Archive hourly IMF/wind/Dst released after hour end, ap/Kp after 3h bin end, F107 after day end.",
            "Revised OMNI archives and assumed availability; not real historical live receipts.",
            "No forecast future solar wind, IMF or CME arrival inputs; future ap is inferred from past observations only.",
            "Linear ap interpolation precedes Kp conversion; Kp has 3h lag and trailing 24h mean.",
            "Metrics use only common valid predictions; missing input coverage reported separately.",
            "Descriptive day-block bootstrap; nearby storm days are correlated.",
            "One satellite, three windows; no claim of generalization to all LEO altitudes or storms.",
        ],
    }
    (OUT/"results.json").write_text(json.dumps(provenance, indent=2)+"\n")
    from report_three_way import write_report
    write_report(OUT)
    print(scores.loc[scores.grouping.eq("lead_hours"), ["lead_hours", "model", "n", "mape_pct", "bias_pct", "underestimate_gt_30pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
