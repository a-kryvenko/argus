"""Prepare model inputs from an explicit observation-service response."""
import pandas as pd
from datetime import datetime
from common.schemas.forecast_inputs import ForecastInputs
from forecast_core.api import (
    prepare_density_drivers,
)


def observed_driver_frame(measurements: pd.DataFrame, issue_time: datetime) -> pd.DataFrame:
    return prepare_density_drivers(measurements, issue_time)


def load_density_drivers(inputs: ForecastInputs, issue_time: datetime) -> pd.DataFrame:
    records = pd.DataFrame(
        [row.model_dump() for row in inputs.measurements],
        columns=['metric', 'value', 'observed_at'],
    )
    return observed_driver_frame(records, issue_time)
