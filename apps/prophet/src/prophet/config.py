"""Operational limits; model mathematics and bundle horizons remain in backends."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator


class InputPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    min_normalized_points: int = Field(default=0, ge=0)
    max_normalized_age_hours: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    def validate_inputs(self, inputs):
        points = inputs.observations.points
        if len(points) < self.min_normalized_points:
            raise ValueError(f'Need at least {self.min_normalized_points} normalized input points')
        if self.max_normalized_age_hours is not None:
            times = [p.issue_time for p in points if p.issue_time.tzinfo is not None]
            age = (inputs.as_of-max(times)).total_seconds()/3600 if times else None
            if age is None or not 0 <= age <= self.max_normalized_age_hours:
                raise ValueError('Normalized input age exceeds configured policy')


class SummarySettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    cme_grace_hours: int = Field(default=24, ge=0, le=168)
    critical_levels: dict[str, Annotated[list[FiniteFloat], Field(min_length=1, max_length=4)]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def ordered_levels(self):
        if any(not (all(a < b for a, b in zip(levels, levels[1:])) or
                    all(a > b for a, b in zip(levels, levels[1:])))
               for levels in self.critical_levels.values()):
            raise ValueError('Critical levels must strictly increase or decrease')
        return self


class ProphetConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    summary: SummarySettings = Field(default_factory=SummarySettings)
    calculation_timeout_seconds: int = Field(default=540, ge=1)
    shutdown_grace_seconds: int = Field(default=570, ge=1, le=570)
    inputs: dict[str, InputPolicy] = Field(default_factory=dict)
    verification_days: int = Field(default=30, ge=1)
    verification_timeout_seconds: int = Field(default=300, ge=1, le=540)

    @field_validator('inputs')
    @classmethod
    def known_products(cls, policies):
        from prophet.services.generation.products import PRODUCTS
        if set(policies)-set(PRODUCTS):
            raise ValueError('Unknown product configuration')
        return policies


def load_config():
    from common.config import get_config
    return ProphetConfig.model_validate(get_config().project_config.get('prophet', {}))
