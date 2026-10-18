import pandas as pd

from common.schemas.forecast import Forecast
from common.schemas.observation import Observation


def observations_to_dataframe(observation: Observation) -> pd.DataFrame:
    return pd.DataFrame([o.model_dump() for o in observation.points])

def forecast_to_dataframe(forecast: Forecast) -> pd.DataFrame:
    df = pd.DataFrame([f.model_dump() for f in forecast.points])
    df.insert(0, "issue_time", forecast.issue_time)
    return df
