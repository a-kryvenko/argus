"""Operational limits; model mathematics and bundle horizons remain in backends."""
from datetime import UTC, datetime, timedelta
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProductSchedule(BaseModel):
    model_config = ConfigDict(extra='forbid')
    every_minutes: int = Field(default=60, ge=1, le=10080)
    offset_minutes: int = Field(default=10, ge=0)

    @model_validator(mode='after')
    def valid_offset(self):
        if self.offset_minutes >= self.every_minutes:
            raise ValueError('Schedule offset must be less than its interval')
        return self

    def due_slot(self, now):
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError('Scheduler time must be timezone-aware')
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        interval = timedelta(minutes=self.every_minutes)
        return epoch + ((now.astimezone(UTC)-timedelta(minutes=self.offset_minutes)-epoch)//interval)*interval


class InputPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    min_normalized_points: int = Field(default=0, ge=0)
    max_normalized_age_hours: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_gong_age_hours: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    def validate_inputs(self, inputs):
        points = inputs.observations.points
        if len(points) < self.min_normalized_points:
            raise ValueError(f'Need at least {self.min_normalized_points} normalized input points')
        if self.max_normalized_age_hours is not None:
            times = [p.issue_time for p in points if p.issue_time.tzinfo is not None]
            age = (inputs.as_of-max(times)).total_seconds()/3600 if times else None
            if age is None or not 0 <= age <= self.max_normalized_age_hours:
                raise ValueError('Normalized input age exceeds configured policy')
        if self.max_gong_age_hours is not None:
            gong = inputs.gong
            age = (inputs.as_of-gong.observed_at).total_seconds()/3600 if gong else None
            if age is None or not 0 <= age <= self.max_gong_age_hours or gong.available_at > inputs.as_of:
                raise ValueError('GONG input is missing, future or exceeds configured age policy')


class VerificationSchedule(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = False
    every_hours: int = Field(default=6, ge=1, le=168)
    days: int = Field(default=7, ge=1, le=25)
    timeout_seconds: int = Field(default=300, ge=1, le=540)


class ProphetConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    calculation_timeout_seconds: int = Field(default=540, ge=1)
    shutdown_grace_seconds: int = Field(default=570, ge=1, le=570)
    inputs: dict[str, InputPolicy] = Field(default_factory=dict)
    verification: VerificationSchedule = Field(default_factory=VerificationSchedule)
    max_parallel_products: int = Field(default=2, ge=1, le=16)
    schedules: dict[str, ProductSchedule] = Field(default_factory=dict)

    @field_validator('inputs', 'schedules')
    @classmethod
    def known_products(cls, policies):
        from argus_prophet.services.generation.products import PRODUCTS
        if set(policies)-set(PRODUCTS):
            raise ValueError('Unknown product configuration')
        return policies


def load_config():
    from common.config import get_config
    return ProphetConfig.model_validate(get_config().project_config.get('prophet', {}))
