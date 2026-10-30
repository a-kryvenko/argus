"""Homepage chart inputs and cache settings."""
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

IMAGE_NAME_PATTERN = r'^[0-9a-f]{64}\.svg$'
Levels = Annotated[list[FiniteFloat], Field(min_length=1, max_length=4)]


def ordered(levels):
    return len(levels) == 1 or all(a < b for a, b in zip(levels, levels[1:])) or all(
        a > b for a, b in zip(levels, levels[1:]))


class ChartPoint(BaseModel):
    model_config = ConfigDict(extra='forbid')
    time: AwareDatetime
    value: FiniteFloat


class ForecastImageData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    before: list[ChartPoint] = Field(min_length=2, max_length=3)
    current: ChartPoint
    after: list[ChartPoint] = Field(min_length=2, max_length=3)
    critical_levels: Levels

    @model_validator(mode='after')
    def check_order(self):
        points = [*self.before, self.current, *self.after]
        if any(a.time >= b.time for a, b in zip(points, points[1:])):
            raise ValueError('Point times must strictly increase through the current observation')
        if not ordered(self.critical_levels):
            raise ValueError('Critical levels must strictly increase or decrease in danger order')
        return self


class SummarySettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ttl_seconds: int = Field(default=600, ge=1, le=600)
    image_ttl_seconds: int = Field(default=1800, ge=1)
    cme_grace_hours: int = Field(default=24, ge=0, le=168)
    critical_levels: dict[str, Levels] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_settings(self):
        if self.image_ttl_seconds <= self.ttl_seconds:
            raise ValueError('Images must outlive the summary cache')
        if any(not ordered(levels) for levels in self.critical_levels.values()):
            raise ValueError('Color levels must strictly increase or decrease in danger order')
        return self
