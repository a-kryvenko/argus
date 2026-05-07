#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


FEATURE_COLUMNS = [
    "lead_hours",
    "lead_norm",

    "v_obs",
    "n_obs",
    "bz_obs",
    "bt_obs",
    "kp",

    "v_persist_1h",
    "v_persist_6h",
    "v_persist_24h",
    "v_persist_27d",

    "delta_v_1h_6h",
    "delta_v_1h_24h",
    "delta_v_24h_27d",

    "abs_bz",
    "southward_bz",
]


QUANTILES = {
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

    needed = ["issue_time", "valid_time", "target_v", *FEATURE_COLUMNS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["target_v", *FEATURE_COLUMNS])
    df = df.sort_values(["issue_time", "lead_hours"])

    return df


def time_split_train_calib_test(df, train_ratio=0.70, calib_ratio=0.15):
    issue_times = np.array(sorted(df["issue_time"].unique()))

    train_end_idx = int(len(issue_times) * train_ratio)
    calib_end_idx = int(len(issue_times) * (train_ratio + calib_ratio))

    train_end = issue_times[train_end_idx]
    calib_end = issue_times[calib_end_idx]

    train = df[df["issue_time"] < train_end].copy()
    calib = df[
        (df["issue_time"] >= train_end)
        & (df["issue_time"] < calib_end)
    ].copy()
    test = df[df["issue_time"] >= calib_end].copy()

    return train, calib, test

def train_quantile_model(train: pd.DataFrame, alpha: float) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["target_v"],
    )

    return model


def fix_quantile_crossing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    q10 = df["pred_q10"].to_numpy()
    q50 = df["pred_q50"].to_numpy()
    q90 = df["pred_q90"].to_numpy()

    ordered = np.sort(np.vstack([q10, q50, q90]), axis=0)

    df["pred_q10"] = ordered[0]
    df["pred_q50"] = ordered[1]
    df["pred_q90"] = ordered[2]

    return df

def evaluate_by_lead(df):
    rows = []

    for lead, g in df.groupby("lead_hours"):
        y = g["target_v"].to_numpy()

        rows.append({
            "lead_hours": int(lead),

            "q50_mae": mean_absolute_error(y, g["pred_q50"]),
            "q50_rmse": root_mean_squared_error(y, g["pred_q50"]),

            "raw_coverage_80": interval_coverage(y, g["pred_q10"], g["pred_q90"]),
            "cal_coverage_80": interval_coverage(y, g["cal_q10"], g["cal_q90"]),

            "raw_width_80": float(np.mean(g["pred_q90"] - g["pred_q10"])),
            "cal_width_80": float(np.mean(g["cal_q90"] - g["cal_q10"])),

            "scale": g["scale"].iloc[0],
            "n": len(g),
        })

    return pd.DataFrame(rows)

def add_quantile_predictions(df, models):
    out = df.copy()

    for name, model in models.items():
        out[f"pred_{name}"] = model.predict(out[FEATURE_COLUMNS])

    # fix quantile crossing
    q = np.sort(
        np.vstack([
            out["pred_q10"].to_numpy(),
            out["pred_q50"].to_numpy(),
            out["pred_q90"].to_numpy(),
        ]),
        axis=0,
    )

    out["pred_q10"] = q[0]
    out["pred_q50"] = q[1]
    out["pred_q90"] = q[2]

    return out

def learn_lead_hour_calibration(df, target_coverage=0.80):
    rows = []

    for lead, g in df.groupby("lead_hours"):
        y = g["target_v"].to_numpy()
        q10 = g["pred_q10"].to_numpy()
        q50 = g["pred_q50"].to_numpy()
        q90 = g["pred_q90"].to_numpy()

        lower_width = np.maximum(q50 - q10, 1e-6)
        upper_width = np.maximum(q90 - q50, 1e-6)

        required_scale = np.maximum(
            np.where(y < q50, (q50 - y) / lower_width, 0.0),
            np.where(y > q50, (y - q50) / upper_width, 0.0),
        )

        scale = np.quantile(required_scale, target_coverage)

        rows.append({
            "lead_hours": int(lead),
            "scale": max(1.0, float(scale)),
            "n": len(g),
        })

    return pd.DataFrame(rows)

def apply_calibration(df, calibration):
    out = df.merge(calibration[["lead_hours", "scale"]], on="lead_hours", how="left")

    out["cal_q50"] = out["pred_q50"]

    out["cal_q10"] = (
        out["pred_q50"]
        - out["scale"] * (out["pred_q50"] - out["pred_q10"])
    )

    out["cal_q90"] = (
        out["pred_q50"]
        + out["scale"] * (out["pred_q90"] - out["pred_q50"])
    )

    return out

def pinball_loss(y_true, y_pred, alpha):
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1.0) * err)))


def interval_coverage(y, q10, q90):
    return float(np.mean((y >= q10) & (y <= q90)))

def evaluate_probabilistic(df):
    y = df["target_v"].to_numpy()

    return pd.DataFrame([
        {
            "metric": "q50_mae",
            "value": mean_absolute_error(y, df["pred_q50"]),
        },
        {
            "metric": "q50_rmse",
            "value": root_mean_squared_error(y, df["pred_q50"]),
        },
        {
            "metric": "q10_pinball",
            "value": pinball_loss(y, df["pred_q10"], 0.10),
        },
        {
            "metric": "q50_pinball",
            "value": pinball_loss(y, df["pred_q50"], 0.50),
        },
        {
            "metric": "q90_pinball",
            "value": pinball_loss(y, df["pred_q90"], 0.90),
        },
        {
            "metric": "raw_coverage_80",
            "value": interval_coverage(y, df["pred_q10"], df["pred_q90"]),
        },
        {
            "metric": "cal_coverage_80",
            "value": interval_coverage(y, df["cal_q10"], df["cal_q90"]),
        },
        {
            "metric": "raw_width_80",
            "value": float(np.mean(df["pred_q90"] - df["pred_q10"])),
        },
        {
            "metric": "cal_width_80",
            "value": float(np.mean(df["cal_q90"] - df["cal_q10"])),
        },
    ])

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train probabilistic solar wind speed quantile models."
    )

    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--model-out",
        default=Path("models/speed_quantile_models.joblib"),
        type=Path,
    )
    parser.add_argument(
        "--predictions-out",
        default=Path("reports/speed_quantile_valid_predictions.csv"),
        type=Path,
    )
    parser.add_argument(
        "--metrics-out",
        default=Path("reports/speed_quantile_metrics.csv"),
        type=Path,
    )
    parser.add_argument(
        "--metrics-by-lead-out",
        default=Path("reports/speed_quantile_metrics_by_lead.csv"),
        type=Path,
    )
    parser.add_argument("--train-ratio", default=0.8, type=float)

    args = parser.parse_args()

    df = load_dataset(args.dataset)
    train, calib, valid = time_split_train_calib_test(df)

    print("train:", len(train), train["issue_time"].min(), "->", train["issue_time"].max())
    print("calib:", len(calib), calib["issue_time"].min(), "->", calib["issue_time"].max())
    print("valid :", len(valid), valid["issue_time"].min(), "->", valid["issue_time"].max())

    models = {}

    for name, alpha in QUANTILES.items():
        print(f"training {name} alpha={alpha}")
        models[name] = train_quantile_model(train, alpha)

    calib_pred = add_quantile_predictions(calib, models)
    valid_pred = add_quantile_predictions(valid, models)

    calibration = learn_lead_hour_calibration(calib_pred, target_coverage=0.80)
    valid_cal = apply_calibration(valid_pred, calibration)

    valid_cal = fix_quantile_crossing(valid_cal)

    metrics = evaluate_probabilistic(valid_cal)
    metrics_by_lead = evaluate_by_lead(valid_cal)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_by_lead_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "models": models,
            "feature_columns": FEATURE_COLUMNS,
            "quantiles": QUANTILES,
        },
        args.model_out,
    )

    valid.to_csv(args.predictions_out, index=False)
    metrics.to_csv(args.metrics_out, index=False)
    metrics_by_lead.to_csv(args.metrics_by_lead_out, index=False)

    print()
    print(metrics)
    print()
    print(f"saved models: {args.model_out}")
    print(f"saved predictions: {args.predictions_out}")
    print(f"saved metrics: {args.metrics_out}")
    print(f"saved lead metrics: {args.metrics_by_lead_out}")


if __name__ == "__main__":
    main()