"""Contract and output conversion for exceedance probability services."""
from abc import ABC, abstractmethod

from common.schemas.forecast import Forecast, ForecastPoint
import pandas as pd


class ProbabilityFS(ABC):
    registry_name: str
    target_name: str
    thresholds: list

    @abstractmethod
    def forecast_snapshot(self, inputs, *, issue_time) -> pd.DataFrame:
        """Predict probabilities from explicitly supplied inputs."""

    def _forecast_row(self, row):
        return {f'p_{self.target_name}_ge_{threshold}': float(row[f'p_{self.target_name}_ge_{threshold}'])
                for threshold in self.thresholds}

    def forecast_from_df(self, frame):
        return Forecast(issue_time=frame.iloc[0].issue_time, points=[
            ForecastPoint(lead_hours=int(row["lead_hours"]), valid_time=row["valid_time"],
                          **self._forecast_row(row)) for row in frame.to_dict('records')])
