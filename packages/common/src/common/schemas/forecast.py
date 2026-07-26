from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ForecastPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    valid_time: datetime
    lead_hours: int

class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_time: datetime
    query: dict = Field(default_factory=dict)
    points: list[ForecastPoint]

