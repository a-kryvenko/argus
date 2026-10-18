"""Contract and output conversion for quantile forecast services."""
from abc import ABC, abstractmethod

from common.schemas.forecast import Forecast, ForecastPoint
import pandas as pd


class QuantileFS(ABC):
    registry_name: str
    target_name: str

    @abstractmethod
    def forecast_snapshot(self, inputs, *, issue_time) -> pd.DataFrame:
        """Predict quantiles from explicitly supplied inputs."""

    def _forecast_row(self, row):
        return {f'{self.target_name}_q{q}': float(row[f'{self.target_name}_q{q}'])
                for q in (10, 50, 90)}

    def forecast_from_df(self, frame):
        return Forecast(issue_time=frame.iloc[0].issue_time, points=[
            ForecastPoint(lead_hours=int(row["lead_hours"]), valid_time=row["valid_time"],
                          **self._forecast_row(row)) for row in frame.to_dict('records')])
