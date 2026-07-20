from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class QuantileForecast(BaseModel):
    q10: float
    q50: float
    q90: float


class BinaryForecast(BaseModel):
    threshold: float
    operator: Literal["gte"] = "gte"
    probability: float


class VariableForecast(BaseModel):
    unit: str
    continuous: QuantileForecast | None = None
    binary: list[BinaryForecast] = Field(default_factory=list)


class ForecastPoint(BaseModel):
    valid_time: datetime
    lead_hours: int
    variables: dict[str, VariableForecast]


class Forecast(BaseModel):
    target: str
    issue_time: datetime
    horizon_hours: int
    available_variables: list[str]
    predictions: list[ForecastPoint]
