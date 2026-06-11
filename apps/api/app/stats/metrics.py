import pandas as pd

from common.config import get_config

def wind_speed_metrics():
    config = get_config()

    df = pd.read_csv(config.workdir / "metrics/wind_speed_by_lead.csv")

    return df.to_dict('records')

def wind_threshold_metrics():
    config = get_config()

    df = pd.read_csv(config.workdir / "metrics/wind_threshold_by_lead.csv")

    reframed_df = (
        df.pivot(
            index="lead_hours",
            columns="threshold",
            values="roc_auc"
        )
        .rename(columns=lambda x: f"roc_auc_{x}")
        .reset_index()
    )

    return reframed_df.to_dict('records')