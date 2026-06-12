#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from forecast.data_pipelines.feature_building import BASE_FEATURE_COLUMNS


FEATURE_COLUMNS = [
    "lead_hours",
    "lead_norm",
] + BASE_FEATURE_COLUMNS


BASELINES = {
    "persist_1h": "v_persist_1h",
    "persist_6h": "v_persist_6h",
    "persist_24h": "v_persist_24h",
    "persist_27d": "v_persist_27d",
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


def time_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    issue_times = np.array(sorted(df["issue_time"].unique()))
    split_idx = int(len(issue_times) * train_ratio)

    train_until = issue_times[split_idx]

    train = df[df["issue_time"] < train_until].copy()
    valid = df[df["issue_time"] >= train_until].copy()

    return train, valid


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }


def evaluate_by_lead(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []

    for lead, g in df.groupby("lead_hours"):
        m = metrics(g["target_v"].values, g[pred_col].values)
        rows.append({
            "model": pred_col,
            "lead_hours": lead,
            **m,
            "n": len(g),
        })

    return pd.DataFrame(rows)


def add_blended_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Simple hand-made blend:
    # short horizon: recent persistence dominates
    # long horizon: 24h and 27d recurrence get more weight
    lead = df["lead_hours"].clip(1, 96)

    w_recent = np.maximum(0.15, 1.0 - lead / 96.0)
    w_24h = 0.25
    w_27d = 1.0 - w_recent - w_24h
    w_27d = np.maximum(0.0, w_27d)

    total = w_recent + w_24h + w_27d

    w_recent = w_recent / total
    w_24h = w_24h / total
    w_27d = w_27d / total

    df["baseline_blend"] = (
        w_recent * df["v_persist_1h"]
        + w_24h * df["v_persist_24h"]
        + w_27d * df["v_persist_27d"]
    )

    return df


def train_lgbm(train: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
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
        eval_set=[(train[FEATURE_COLUMNS], train["target_v"])],
        eval_metric="l1",
    )

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train persistence baselines and LightGBM solar wind speed model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Training dataset CSV created from observation features.",
        default=Path("data/historical/omni_processed.csv")
    )
    parser.add_argument(
        "--model-out",
        default=Path("data/models/speed_lgbm.joblib"),
        type=Path,
    )
    parser.add_argument(
        "--metrics-out",
        default=Path("data/reports/speed_baseline_metrics_by_lead.csv"),
        type=Path,
    )
    parser.add_argument(
        "--train-ratio",
        default=0.8,
        type=float,
    )

    args = parser.parse_args()

    df = load_dataset(args.dataset)
    df = add_blended_baseline(df)

    train, valid = time_split(df, args.train_ratio)

    print(f"train rows: {len(train)}")
    print(f"valid rows: {len(valid)}")
    print(f"train issue range: {train['issue_time'].min()} -> {train['issue_time'].max()}")
    print(f"valid issue range: {valid['issue_time'].min()} -> {valid['issue_time'].max()}")

    results = []

    for name, col in BASELINES.items():
        valid[f"pred_{name}"] = valid[col]
        m = metrics(valid["target_v"].values, valid[f"pred_{name}"].values)
        print(f"{name}: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, Bias={m['bias']:.2f}")
        results.append(evaluate_by_lead(valid, f"pred_{name}"))

    valid["pred_blend"] = valid["baseline_blend"]
    m = metrics(valid["target_v"].values, valid["pred_blend"].values)
    print(f"blend: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, Bias={m['bias']:.2f}")
    results.append(evaluate_by_lead(valid, "pred_blend"))

    model = train_lgbm(train)

    valid["pred_lgbm"] = model.predict(valid[FEATURE_COLUMNS])
    m = metrics(valid["target_v"].values, valid["pred_lgbm"].values)
    print(f"lgbm: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, Bias={m['bias']:.2f}")
    results.append(evaluate_by_lead(valid, "pred_lgbm"))

    metrics_by_lead = pd.concat(results, ignore_index=True)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        args.model_out,
    )

    metrics_by_lead.to_csv(args.metrics_out, index=False)

    print(f"saved model: {args.model_out}")
    print(f"saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()