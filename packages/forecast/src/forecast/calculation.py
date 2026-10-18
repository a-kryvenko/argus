"""In-memory forecast calculations, independent of storage and configuration."""
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from typing import Protocol
from forecast.inputs import ForecastInputs


@dataclass
class ForecastResult:
    name: str
    frame: pd.DataFrame
    model_info: dict


class SnapshotService(Protocol):
    registry_name: str

    def forecast_snapshot(self, inputs: ForecastInputs, *, issue_time: datetime) -> pd.DataFrame: ...


def calculate_snapshot(service: SnapshotService, inputs: ForecastInputs, *,
                       issue_time: datetime, model_info: dict) -> ForecastResult:
    """Every backend consumes the same saved inputs without storage or configuration."""
    return ForecastResult(service.registry_name,
                          service.forecast_snapshot(inputs, issue_time=issue_time), model_info)
