"""Convert provider datasets into storage records."""
import pandas as pd

from clio.observations.schema import MEASUREMENT_METRICS


def wide_to_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in (*MEASUREMENT_METRICS, 'bt') if metric in frame]
    return (frame.melt(id_vars='timestamp', value_vars=metrics, var_name='metric', value_name='value')
            .rename(columns={'timestamp': 'observed_at'}).dropna(subset=['observed_at', 'value']))
