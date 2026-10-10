"""Empirical multiplicative density calibration on a separate chronological set.

Marginal upper/interval coverage is measured on held-out dates, not guaranteed:
time dependence and quiet-to-storm distribution shift violate iid assumptions.
Upper ap scenarios are not density quantiles.
"""
import numpy as np
import pandas as pd


def fit_density_calibration(calibration, earliest_test_issue):
    required = ["truth", "rho_q50"]
    if calibration.empty or calibration.valid_time.max() >= pd.Timestamp(earliest_test_issue):
        raise ValueError("Calibration must precede all test issuances")
    if not np.isfinite(calibration[required]).all().all() or not calibration[required].gt(0).all().all():
        raise ValueError("Calibration requires finite positive densities")
    result = []
    for lead, frame in calibration.groupby("lead_hours"):
        residual = np.log(frame.truth/frame.rho_q50).to_numpy()
        if len(residual) < 100:
            raise ValueError("At least 100 calibration points required per horizon")
        result.append({"lead_hours": int(lead), "n": len(frame),
            "days": int(frame.valid_time.dt.floor("D").nunique()),
            "lower05_factor": float(np.exp(np.quantile(residual, .05, method="lower"))),
            "upper90_factor": float(np.exp(np.quantile(residual, .90, method="higher"))),
            "upper95_factor": float(np.exp(np.quantile(residual, .95, method="higher"))),
            "calibration_start": frame.valid_time.min().isoformat(),
            "calibration_end": frame.valid_time.max().isoformat()})
    return pd.DataFrame(result)


def apply_density_calibration(predictions, parameters):
    result = predictions.merge(parameters[["lead_hours", "lower05_factor", "upper90_factor", "upper95_factor"]],
                               on="lead_hours", how="left", validate="many_to_one")
    if result[["lower05_factor", "upper90_factor", "upper95_factor"]].isna().any().any():
        raise ValueError("Missing calibration horizon")
    result["density_lower05"] = result.rho_q50*result.lower05_factor
    result["density_upper90"] = result.rho_q50*result.upper90_factor
    result["density_upper95"] = result.rho_q50*result.upper95_factor
    return result


def score_bounds(frame):
    output = []
    for grouping in [("lead_hours",), ("lead_hours", "activity")]:
        for key, group in frame.groupby(list(grouping)):
            labels = dict(zip(grouping, key if isinstance(key, tuple) else (key,)))
            for name in ["rho_q50", "rho_q90", "rho_q95", "density_upper90", "density_upper95"]:
                ratio = group[name]/group.truth
                output.append({"grouping": "+".join(grouping), **labels, "scenario": name,
                    "n": len(group), "coverage_pct": float(100*(ratio >= 1).mean()),
                    "underestimate_gt_30_pct": float(100*(ratio < .7).mean()),
                    "overestimate_gt_50_pct": float(100*(ratio > 1.5).mean()),
                    "mape_pct": float(100*(ratio-1).abs().mean()),
                    "median_bound_over_truth": float(ratio.median()),
                    "mean_bound_over_point": float((group[name]/group.rho_q50).mean())})
    return pd.DataFrame(output)


def score_intervals(frame):
    rows = []
    for grouping in [("lead_hours",), ("lead_hours", "activity")]:
        for key, group in frame.groupby(list(grouping)):
            labels = dict(zip(grouping, key if isinstance(key, tuple) else (key,)))
            covered = group.truth.between(group.density_lower05, group.density_upper95)
            # 90% interval score penalizes both excess width and misses.
            lower, upper, truth = group.density_lower05, group.density_upper95, group.truth
            score = upper-lower+20*np.maximum(lower-truth, 0)+20*np.maximum(truth-upper, 0)
            rows.append({"grouping": "+".join(grouping), **labels, "n": len(group),
                "coverage90_pct": float(100*covered.mean()),
                "below_lower_pct": float(100*(truth < lower).mean()),
                "above_upper_pct": float(100*(truth > upper).mean()),
                "mean_width_over_point": float(((upper-lower)/group.rho_q50).mean()),
                "mean_normalized_interval_score": float((score/truth).mean())})
    return pd.DataFrame(rows)
