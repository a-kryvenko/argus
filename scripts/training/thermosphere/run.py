"""Paired GRACE-FO density benchmark of current JB2008 and DTM2020 Operational.

Run: .venv/bin/python -u scripts/training/thermosphere/run.py
Inputs: prepare.py downloads, existing OMNI/GOES archives and frozen calibration.
This is a retrospective, closed-bin persistence experiment, not a live replay.
Future measured positions are shared conditioning coordinates, not predicted orbits.
"""
import json
from pathlib import Path
import subprocess
import time

import cdflib
import numpy as np
import pandas as pd

from prepare import CONFIG, HERE, OUT, ROOT, digest
from models import DTM2020, dtm_snapshot, jb_evaluate, jb_geometry


def load_observations(windows=None, directory=None):
    windows = CONFIG["windows"] if windows is None else windows
    directory = OUT / "gracefo" if directory is None else Path(directory)
    frames, coverage = [], []
    expected = [d for start, end in windows for d in pd.date_range(start, end)]
    for day in expected:
        stamp = day.strftime("%Y%m%d")
        name = f"GF_OPER_DNS1ACC_2__{stamp}T000000_{stamp}T235959_{CONFIG['density_product_version']}.cdf"
        path = directory / name
        cdf = cdflib.CDF(path)
        frame = pd.DataFrame({
            "valid_time": pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget("time")), utc=True),
            "altitude_km": cdf.varget("altitude")/1000,
            "longitude_deg": cdf.varget("longitude"), "latitude_deg": cdf.varget("latitude"),
            "local_solar_time": cdf.varget("local_solar_time"),
            "truth": cdf.varget("density"), "quality": cdf.varget("validity_flag"),
        })
        for variable, unit in [("altitude", "m"), ("density", "kg/m3"),
                               ("latitude", "deg"), ("longitude", "deg")]:
            if cdf.varattsget(variable)["UNITS"] != unit:
                raise ValueError(f"Unexpected {variable} units")
        # Select nearest raw sample to each predeclared grid point BEFORE quality
        # filtering. Never replace a flagged sample with a later good observation.
        grid = pd.DataFrame({"target_time": pd.date_range(day, day+pd.Timedelta(days=1),
                            freq=f"{CONFIG['sample_minutes']}min", inclusive="left", tz="UTC")})
        sampled = pd.merge_asof(grid, frame.sort_values("valid_time"), left_on="target_time",
                                right_on="valid_time", direction="nearest",
                                tolerance=pd.Timedelta(seconds=5))
        valid = (sampled.quality.eq(0) & sampled.truth.between(1e-16, 1e-8)
                 & sampled.altitude_km.between(120.001, 1500)
                 & sampled.latitude_deg.between(-90, 90)
                 & sampled.longitude_deg.between(-180, 180)
                 & sampled.local_solar_time.between(0, 24)
                 & sampled.valid_time.notna())
        coverage.append({"date": str(day.date()), "raw_rows": len(frame),
                         "requested": len(sampled), "valid": int(valid.sum())})
        sampled = sampled.loc[valid].copy()
        sampled["window"] = next(start for start, end in windows if pd.Timestamp(start) <= day <= pd.Timestamp(end))
        frames.append(sampled)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coverage)


def load_measurements():
    from forecast_core.data_pipelines.solar_indices import extract_solar_indices, load_solar_index_calibrations
    start = pd.Timestamp(CONFIG["windows"][0][0], tz="UTC")-pd.Timedelta(days=95)
    end = pd.Timestamp(CONFIG["windows"][-1][1], tz="UTC")+pd.Timedelta(days=1)
    files = [ROOT / f"data/raw/omni_{split}.csv" for split in ("2010_2024", "2025")]
    omni = pd.concat([pd.read_csv(f, usecols=["issue_time", "f10_7", "ap", "kp", "dst"])
                      for f in files], ignore_index=True)
    omni["issue_time"] = pd.to_datetime(omni.issue_time, utc=True)
    omni = omni.loc[omni.issue_time.between(start, end)].set_index("issue_time").sort_index()
    if omni.index.has_duplicates:
        raise ValueError("Duplicate OMNI timestamps")
    frames = []
    for metric, frequency, shift, delay, bounds in [
        ("f10_7", "D", pd.Timedelta(hours=12), pd.Timedelta(hours=12), (30, 500)),
        ("ap", "3h", pd.Timedelta(0), pd.Timedelta(hours=3), (0, 400)),
        ("dst", "h", pd.Timedelta(0), pd.Timedelta(hours=1), (-2000, 100)),
    ]:
        # F107 and ap repeat within their bins in OMNI. Reject inconsistent bins.
        groups = omni[metric].resample(frequency)
        if (groups.nunique() > 1).any():
            raise ValueError(f"Inconsistent {metric} archive bins")
        series = groups.first()
        series = series.where(series.between(*bounds))
        f = pd.DataFrame({"observed_at": series.index+shift, "value": series.values, "metric": metric})
        f["available_at"] = f.observed_at+delay
        frames.append(f)
    calibration = ROOT / "data/models/argus-solar-indices-cal-v1.json"
    models = load_solar_index_calibrations(calibration)
    for split in ("2010_2024", "2025"):
        path = ROOT / f"data/raw/goes/goes_{split}.parquet"
        files.append(path)
        goes = pd.read_parquet(path, filters=[("timestamp", ">=", start), ("timestamp", "<", end)])
        daily = extract_solar_indices(goes, models)
        f = daily.melt(id_vars="timestamp", value_vars=["s10", "m10", "y10"],
                       var_name="metric", value_name="value").rename(columns={"timestamp": "observed_at"})
        f["available_at"] = f.observed_at.dt.floor("D")+pd.Timedelta(days=1)
        frames.append(f)
    files.append(calibration)
    measurements = pd.concat(frames, ignore_index=True).sort_values("observed_at")
    return measurements, omni, files


def paired_metrics(frame):
    rows = []
    for model in ("jb2008", "dtm2020"):
        truth, prediction = frame.truth.to_numpy(), frame[model].to_numpy()
        ratio = prediction/truth
        rows.append({"model": model, "n": len(frame), "days": frame.valid_time.dt.floor("D").nunique(),
                     "mae_kg_m3": float(np.abs(prediction-truth).mean()),
                     "rmse_kg_m3": float(np.sqrt(np.mean((prediction-truth)**2))),
                     "mape_pct": float(100*np.abs(ratio-1).mean()),
                     "median_ape_pct": float(100*np.median(np.abs(ratio-1))),
                     "bias_pct": float(100*(ratio-1).mean()),
                     "log_rmse": float(np.sqrt(np.mean(np.log(ratio)**2))),
                     "underestimate_gt_30pct": float(np.mean(ratio < .7)),
                     "correlation": float(np.corrcoef(truth, prediction)[0, 1]) if len(frame)>2 and np.std(truth)>0 and np.std(prediction)>0 else None})
    return rows


def summarize(pairs):
    output = []
    for grouping in [("lead_hours",), ("lead_hours", "activity"), ("lead_hours", "window")]:
        for key, group in pairs.groupby(list(grouping)):
            key = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(grouping, key))
            for row in paired_metrics(group):
                output.append({"grouping": "+".join(grouping), **labels, **row})
    return pd.DataFrame(output)


def bootstrap(pairs):
    rng = np.random.default_rng(CONFIG["seed"])
    result = []
    for lead, frame in pairs.groupby("lead_hours"):
        frame = frame.copy()
        frame["delta"] = 100*((frame.dtm2020/frame.truth-1).abs()-(frame.jb2008/frame.truth-1).abs())
        daily = frame.groupby(frame.valid_time.dt.floor("D")).delta.agg(["sum", "count"])
        selections = rng.integers(0, len(daily), size=(2000, len(daily)))
        means = daily["sum"].to_numpy()[selections].sum(axis=1)/daily["count"].to_numpy()[selections].sum(axis=1)
        result.append({"lead_hours": int(lead), "dtm_minus_jb_mape_pp": float(frame.delta.mean()),
                       "day_block_ci95_low": float(np.quantile(means, .025)),
                       "day_block_ci95_high": float(np.quantile(means, .975))})
    return result


def main():
    from forecast_core.models.jb2008_drivers import prepare_snapshot, DriverDataUnavailable
    began = time.monotonic()
    for artifact in json.loads((OUT / "download_manifest.json").read_text()):
        if digest(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Input changed since download: {artifact['path']}")
    build = json.loads((OUT / "build.json").read_text())
    if (digest(OUT / "libdtm2020.so") != build["library_sha256"]
            or digest(HERE / "dtm_bridge.f90") != build["bridge_sha256"]):
        raise ValueError("DTM build changed; rerun prepare.py")
    observations, coverage = load_observations()
    measurements, omni, input_files = load_measurements()
    measurements.to_parquet(OUT / "measurements.parquet", index=False)
    observations.to_parquet(OUT / "observations.parquet", index=False)
    coverage.to_csv(OUT / "observation_coverage.csv", index=False)
    dtm = DTM2020(OUT)
    # Published four-point benchmark, including the g/cm3 -> kg/m3 conversion.
    actual = [dtm.evaluate(180, alt, 0, 0, 3.1415, flux, flux, 3, 3)[0]
              for alt in (300, 800) for flux in (80, 180)]
    expected = np.array([.94719e-14, .29053e-13, .26600e-17, .39104e-16])*1000
    np.testing.assert_allclose(actual, expected, rtol=5e-5)
    (OUT / "dtm_benchmark.json").write_text(json.dumps({"actual_kg_m3": actual,
        "reference_kg_m3": expected.tolist(), "rtol": 5e-5, "passed": True}, indent=2)+"\n")
    geometry = jb_geometry(observations)
    cache, driver_records, rows = {}, [], []
    print(f"Scoring {len(observations)} observations, {len(CONFIG['lead_hours'])} leads", flush=True)
    for i, observation in enumerate(observations.itertuples(index=False)):
        # Round down to a completed issuance boundary, including a nearest sample
        # just before the nominal target. Record actual lead, never use future inputs.
        boundary = min(observation.target_time, observation.valid_time.floor("30min"))
        activity_kp = omni.kp.get(observation.valid_time.floor("h"), np.nan)
        activity = "unknown" if not np.isfinite(activity_kp) else "storm" if activity_kp >= 5 else "moderate" if activity_kp >= 3 else "quiet"
        for lead in CONFIG["lead_hours"]:
            issue = boundary-pd.Timedelta(hours=lead)
            if issue not in cache:
                known = measurements.loc[(measurements.available_at <= issue)
                        & (measurements.observed_at <= issue)
                        & (measurements.observed_at >= issue-pd.Timedelta(days=90))]
                values = {}
                for model, prepare in [("jb2008", lambda: prepare_snapshot(known, issue).values),
                                       ("dtm2020", lambda: dtm_snapshot(known, issue))]:
                    try:
                        values[model] = prepare()
                    except (DriverDataUnavailable, ValueError) as error:
                        values[model] = None
                        values[model+"_error"] = str(error)
                cache[issue] = values
                driver_records.append({"issue_time": issue.isoformat(), **values})
            state = cache[issue]
            row = {"valid_time": observation.valid_time, "issue_time": issue,
                   "lead_hours": lead, "actual_lead_hours": (observation.valid_time-issue).total_seconds()/3600,
                   "window": observation.window, "activity": activity, "kp_at_valid_time": activity_kp,
                   "latitude_deg": observation.latitude_deg, "longitude_deg": observation.longitude_deg,
                   "altitude_km": observation.altitude_km, "truth": observation.truth}
            for model in ("jb2008", "dtm2020"):
                row[model] = np.nan
                row[model+"_error"] = state.get(model+"_error", "")
                if state[model] is None:
                    continue
                if model == "jb2008":
                    row[model] = jb_evaluate(geometry[i], state[model])
                else:
                    d = state[model]
                    row[model] = float(dtm.evaluate(geometry[i, 1], observation.altitude_km,
                        np.deg2rad(observation.latitude_deg), np.deg2rad(observation.longitude_deg),
                        observation.local_solar_time*np.pi/12,
                        d["f107"], d["f107_mean"], d["kp"], d["kp_mean"])[0])
            rows.append(row)
        if i % 200 == 0:
            print(f"{i}/{len(observations)} observations; {time.monotonic()-began:.0f}s", flush=True)
    predictions = pd.DataFrame(rows)
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    pairs = predictions.dropna(subset=["jb2008", "dtm2020"])
    if pairs.empty:
        raise RuntimeError("No paired predictions; inspect predictions.parquet")
    metrics = summarize(pairs)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    available = predictions.groupby("lead_hours").agg(requested=("truth", "size"),
                jb2008_available=("jb2008", "count"), dtm2020_available=("dtm2020", "count"))
    available["paired"] = pairs.groupby("lead_hours").size()
    available.to_csv(OUT / "prediction_coverage.csv")
    (OUT / "driver_snapshots.json").write_text(json.dumps(driver_records, indent=2)+"\n")
    provenance = {
        "config": CONFIG, "elapsed_seconds": time.monotonic()-began,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in input_files],
        "scripts": {p.name: digest(p) for p in HERE.glob("*.py")},
        "download_manifest_sha256": digest(OUT / "download_manifest.json"),
        "build_manifest_sha256": digest(OUT / "build.json"),
        "forecast_core_revision": subprocess.check_output(["git", "-C", str(ROOT/"packages/forecast-core"), "rev-parse", "HEAD"], text=True).strip(),
        "production_source_hashes": {name: digest(ROOT/"packages/forecast-core/src/forecast_core"/name) for name in [
            "models/jb2008.py", "models/jb2008_drivers.py", "data_pipelines/solar_indices.py"]},
        "bootstrap": bootstrap(pairs),
        "limitations": ["Revised archive drivers with assumed closed-bin availability; not historical live receipts.",
            "GOES science archive reconstructed by frozen pre-2025 calibration; no 2025 fitting.",
            "Both models persist issue-time drivers; observed future coordinates condition both models.",
            "DTM uses causal interpolation of available ap/F107, holding latest unavailable endpoint; trailing 81-day F107 mean.",
            "JB uses unchanged production causal DTC, lagged trailing means and reconstructed solar indices.",
            "Quality-filtered point densities from one satellite and three selected windows; not global LEO validation.",
            "Day-block bootstrap is descriptive; consecutive storm days remain correlated.",
            "This compares complete driver/model chains, not intrinsic model quality with perfect inputs.",
            "No fitted bias correction, drag coefficient, orbital propagation or risk thresholds evaluated.",
            "DTM academic non-commercial license; no production replacement performed."],
    }
    (OUT / "results.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(metrics.loc[metrics.grouping.eq("lead_hours")].to_string(index=False), flush=True)
    from report import write_report
    write_report(OUT)


if __name__ == "__main__":
    main()
