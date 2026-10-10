"""Evaluate upper ap scenarios and independently calibrated density intervals.

Requires prepare_uncertainty.py --download --train. Calibration parameters are
written from December 2024 before evaluation of the previously examined 2025 set.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from prepare import HERE, ROOT, digest
from prepare_uncertainty import CONFIG, OUT, MEDIAN_DIR
from train_ap import read_omni, causal_features, forecast_ap
from run import load_observations
from run_three_way import BASE, check_baseline
from models import DTM2020, dtm_snapshot
from ap_forecast import forecast_dtm_snapshot
from density_uncertainty import fit_density_calibration, apply_density_calibration, score_bounds, score_intervals

QUANTILES = ("q50", "q90", "q95")


def measurement_history(raw):
    frames = []
    for metric, frequency, shift, delay in [("ap", "3h", 0, 3), ("f10_7", "D", 12, 12)]:
        series = raw[metric].resample(frequency).first()
        times = series.index+pd.Timedelta(hours=shift)
        frames.append(pd.DataFrame({"observed_at": times, "available_at": times+pd.Timedelta(hours=delay),
                                   "metric": metric, "value": series.to_numpy()}))
    return pd.concat(frames, ignore_index=True).sort_values("observed_at")


def make_scenarios(bundles, features, issues):
    frames, missing = [], []
    for name in QUANTILES:
        predictions, gaps = forecast_ap(bundles[name], features, issues)
        frames.append(predictions.rename(columns={"ap_q50": f"ap_{name}"}).set_index(["issue_time", "valid_time", "lead_hours"]))
        missing.append(gaps.assign(quantile=name))
    frame = pd.concat(frames, axis=1).reset_index()
    columns = [f"ap_{name}" for name in QUANTILES]
    before = frame[columns].to_numpy()
    info = {"rows": len(frame), "q90_below_q50": int((before[:, 1] < before[:, 0]).sum()),
            "q95_below_q90": int((before[:, 2] < before[:, 1]).sum()),
            "out_of_range_values": int(((before < 0) | (before > 400)).sum())}
    # Keep the already-evaluated median. Raise crossing upper scenarios only;
    # never lower q50 or reinterpret scenarios as calibrated density quantiles.
    frame[columns] = np.maximum.accumulate(np.clip(before, 0, 400), axis=1)
    return frame, pd.concat(missing, ignore_index=True), info


def evaluate(observations, scenarios, measurements, raw, dtm, *, fixed=None, tag):
    groups = {issue: group for issue, group in scenarios.groupby("issue_time")}
    cache, rows = {}, []
    started = time.monotonic()
    for number, obs in enumerate(observations.itertuples(index=False)):
        for lead in CONFIG["lead_hours"]:
            issue = obs.valid_time-pd.Timedelta(hours=lead)
            row = {"valid_time": obs.valid_time, "issue_time": issue, "lead_hours": lead,
                "truth": obs.truth, "window": obs.window, "altitude_km": obs.altitude_km,
                "latitude_deg": obs.latitude_deg, "longitude_deg": obs.longitude_deg}
            kp = raw.kp.get(obs.valid_time.floor("h"), np.nan)
            row["kp_at_valid_time"] = kp
            row["activity"] = "unknown" if not np.isfinite(kp) else "storm" if kp >= 5 else "moderate" if kp >= 3 else "quiet"
            row.update({f"rho_{q}": np.nan for q in QUANTILES})
            if issue not in groups:
                row["error"] = "missing_ap_model_features"
                rows.append(row)
                continue
            try:
                if issue not in cache:
                    known = measurements.loc[measurements.available_at.le(issue)
                        & measurements.observed_at.ge(issue-pd.Timedelta(days=90))]
                    baseline = fixed[issue] if fixed is not None else dtm_snapshot(known, issue)
                    cache[issue] = (known.loc[known.metric.eq("ap")], baseline)
                known, baseline = cache[issue]
                day = obs.valid_time.dayofyear+(obs.valid_time-obs.valid_time.floor("D")).total_seconds()/86400
                for q in QUANTILES:
                    drivers = forecast_dtm_snapshot(known, groups[issue], issue, obs.valid_time,
                                                   quantile=q, baseline=baseline)
                    row[f"rho_{q}"] = float(dtm.evaluate(day, obs.altitude_km, np.deg2rad(obs.latitude_deg),
                        np.deg2rad(obs.longitude_deg), obs.local_solar_time*np.pi/12,
                        drivers["f107"], drivers["f107_mean"], drivers["kp"], drivers["kp_mean"])[0])
                row["error"] = ""
            except ValueError as error:
                row["error"] = str(error)
            rows.append(row)
        if number % 250 == 0:
            print(f"{tag}: {number}/{len(observations)} positions, {time.monotonic()-started:.0f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT/f"{tag}_all_predictions.parquet", index=False)
    valid = frame.dropna(subset=[f"rho_{q}" for q in QUANTILES]).copy()
    coverage = frame.groupby("lead_hours").size().rename("requested").to_frame()
    coverage["common"] = valid.groupby("lead_hours").size()
    coverage.to_csv(OUT/f"{tag}_coverage.csv")
    if coverage.common.isna().any() or coverage.common.eq(0).any():
        raise ValueError("No scenario predictions for at least one horizon")
    return valid


def main():
    started = time.monotonic()
    check_baseline()
    training = json.loads((OUT/"training.json").read_text())
    if training["config"] != CONFIG:
        raise ValueError("Upper model config changed")
    for item in json.loads((OUT/"download_manifest.json").read_text()):
        if digest(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Calibration data changed")
    if digest(MEDIAN_DIR/"ap_q50.joblib") != training["median_artifact_sha256"]:
        raise ValueError("Median model changed")
    bundles = {"q50": joblib.load(MEDIAN_DIR/"ap_q50.joblib")}
    for q in ("q90", "q95"):
        if digest(OUT/f"ap_{q}.joblib") != training["models"][q]["artifact_sha256"]:
            raise ValueError("Upper model changed")
        bundles[q] = joblib.load(OUT/f"ap_{q}.joblib")
    raw, paths = read_omni()
    for path in paths:
        if digest(path) != training["source_files"][str(path.relative_to(ROOT))]:
            raise ValueError("Driver source changed")
    features = causal_features(raw)
    calibration, coverage = load_observations(CONFIG["calibration_windows"], OUT/"gracefo")
    coverage.to_csv(OUT/"calibration_observation_coverage.csv", index=False)
    test = pd.read_parquet(BASE/"observations.parquet")
    issue_sets = {name: pd.DatetimeIndex(sorted({t-pd.Timedelta(hours=h) for t in frame.valid_time
        for h in CONFIG["lead_hours"]})) for name, frame in [("calibration", calibration), ("test", test)]}
    if calibration.valid_time.max() >= issue_sets["test"].min():
        raise ValueError("Calibration overlaps test issuance")
    all_issues = issue_sets["calibration"].union(issue_sets["test"])
    scenarios, gaps, repairs = make_scenarios(bundles, features, all_issues)
    scenarios.to_parquet(OUT/"ap_scenarios.parquet", index=False)
    gaps.to_csv(OUT/"ap_missing.csv", index=False)
    # Density calibration contains no test observations and is frozen before
    # the test density evaluation below. Only the schema and dates are inspected.
    history = measurement_history(raw.loc[raw.index >= pd.Timestamp("2024-09-01", tz="UTC")])
    dtm = DTM2020(BASE)
    cal = evaluate(calibration, scenarios.loc[scenarios.issue_time.isin(issue_sets["calibration"])],
                   history, raw, dtm, tag="calibration")
    parameters = fit_density_calibration(cal, issue_sets["test"].min())
    parameters.to_csv(OUT/"density_calibration.csv", index=False)
    cal = apply_density_calibration(cal, parameters)
    cal.to_parquet(OUT/"calibration_predictions.parquet", index=False)
    score_bounds(cal).to_csv(OUT/"calibration_bound_metrics.csv", index=False)
    fixed = {pd.Timestamp(row["issue_time"]): row["dtm2020"]
        for row in json.loads((BASE/"driver_snapshots.json").read_text())}
    evaluated = evaluate(test, scenarios.loc[scenarios.issue_time.isin(issue_sets["test"])],
                         history, raw, dtm, fixed=fixed, tag="test")
    # Re-evaluated q50 must agree with the previous three-way experiment.
    previous = pd.read_parquet(MEDIAN_DIR/"predictions.parquet")
    audit = evaluated.merge(previous[["issue_time", "valid_time", "lead_hours", "dtm2020_ap_forecast"]],
                            on=["issue_time", "valid_time", "lead_hours"], validate="one_to_one")
    if len(audit) != len(evaluated):
        raise ValueError("Missing median reference rows")
    np.testing.assert_allclose(audit.rho_q50, audit.dtm2020_ap_forecast, rtol=1e-12, atol=0)
    evaluated = apply_density_calibration(evaluated, parameters)
    evaluated.to_parquet(OUT/"test_predictions.parquet", index=False)
    metrics = score_bounds(evaluated)
    metrics.to_csv(OUT/"bound_metrics.csv", index=False)
    score_intervals(evaluated).to_csv(OUT/"interval_metrics.csv", index=False)
    # Check ap marginal exceedances separately: upper driver scenarios do not
    # automatically inherit calibrated density coverage through a nonlinear model.
    ap_test = scenarios.loc[scenarios.issue_time.isin(issue_sets["test"])].copy()
    ap_test["truth"] = raw.ap.reindex(pd.DatetimeIndex(ap_test.valid_time).floor("h")).to_numpy()
    ap_rows = []
    for lead, group in ap_test.dropna(subset="truth").groupby("lead_hours"):
        for q in QUANTILES:
            ap_rows.append({"lead_hours": int(lead), "quantile": q, "n": len(group),
                           "coverage_pct": float(100*(group.truth <= group[f"ap_{q}"]).mean())})
    pd.DataFrame(ap_rows).to_csv(OUT/"ap_coverage.csv", index=False)
    summary = {"config": CONFIG, "elapsed_seconds": time.monotonic()-started,
        "calibration_min_issue": issue_sets["calibration"].min().isoformat(),
        "calibration_max_valid_time": calibration.valid_time.max().isoformat(),
        "test_min_issue": issue_sets["test"].min().isoformat(),
        "calibration_activity_counts": cal.loc[cal.lead_hours.eq(24)].activity.value_counts().to_dict(),
        "calibration_max_kp": float(cal.kp_at_valid_time.max()), "ap_scenario_repairs": repairs,
        "density_scenario_nonmonotone_rows": int(((evaluated.rho_q90 < evaluated.rho_q50)
            | (evaluated.rho_q95 < evaluated.rho_q90)).sum()),
        "training_sha256": digest(OUT/"training.json"), "median_matches_v2": True,
        "input_hashes": {"v2_predictions": digest(MEDIAN_DIR/"predictions.parquet"),
            "v1_observations": digest(BASE/"observations.parquet"),
            "v1_drivers": digest(BASE/"driver_snapshots.json"),
            "dtm_library": digest(BASE/"libdtm2020.so"),
            "download_manifest": digest(OUT/"download_manifest.json")},
        "source_hashes": {path.name: digest(path) for path in HERE.glob("*.py")},
        "output_hashes": {name: digest(OUT/name) for name in ["density_calibration.csv", "test_predictions.parquet", "bound_metrics.csv"]},
        "limitations": [
            "Upper ap q90/q95 are sensitivity trajectories, not calibrated density quantiles or coherent joint trajectory samples.",
            "Density residual calibration is marginal per horizon; no storm-specific calibration or iid/time-series coverage guarantee.",
            "December 2024 calibration is predominantly quiet; it does not represent severe storms.",
            "Same 2025 test windows were inspected in earlier experiments: this is follow-up evaluation, not a fresh blind holdout.",
            "Calibration precedes test by observation dates; archival products may have been published/revised later. Not a live replay.",
            "F107 remains fixed, future satellite coordinates observed, one satellite near 450–500 km.",
            "Density errors and bound exceedances are not individual satellite loss probabilities or operational alarm false-positive rates.",
        ]}
    (OUT/"results.json").write_text(json.dumps(summary, indent=2)+"\n")
    from report_uncertainty import write_report
    write_report(OUT)
    print(metrics.loc[metrics.lead_hours.eq(24)].to_string(index=False))


if __name__ == "__main__":
    main()
