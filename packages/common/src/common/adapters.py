import pandas as pd
from common.schema import Observation, Forecast

def observations_to_dataframe(observation: Observation) -> pd.DataFrame:
    return pd.DataFrame([o.model_dump() for o in observation.points])

def forecast_to_dataframe(forecast: Forecast) -> pd.DataFrame:
    return pd.DataFrame([f.model_dump() for f in forecast.points])

def forecast_from_dataframe(df: pd.DataFrame) -> Forecast:
    pass