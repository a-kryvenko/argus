import pandas as pd

from common.config import get_config

def wind_speed_metrics():
    config = get_config()

    p = config.workdir / config.project_config["metrics"]["solar_wind_speed"]

    df = pd.read_csv(p)

    return df.to_dict('records')

def wind_threshold_metrics():
    config = get_config()

    p = config.workdir / config.project_config["metrics"]["solar_wind_probability"]

    df = pd.read_csv(p)

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