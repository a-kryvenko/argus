"""Validated observation policies for the incremental ingestion migration."""
from datetime import timedelta
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clio.ingestion.products import OBSERVATIONS, PRODUCTS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Sources(StrictModel):
    live: list[str]
    historical: list[str]

    @field_validator('live', 'historical')
    @classmethod
    def unique_products(cls, values):
        if len(values) != len(set(values)):
            raise ValueError('Source priorities must not contain duplicates')
        return values


class Schedule(StrictModel):
    every: Annotated[timedelta, Field(gt=timedelta(0))]

    @field_validator('every', mode='before')
    @classmethod
    def duration(cls, value):
        import re
        if not isinstance(value, str) or not re.fullmatch(r'[1-9][0-9]*[smhd]', value):
            raise ValueError('Use a positive duration such as 60s, 5m, 6h or 1d')
        return timedelta(seconds=int(value[:-1]) * {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[value[-1]])


class Schedules(StrictModel):
    live: Schedule
    backfill: Schedule


class Backfill(StrictModel):
    days: Annotated[int, Field(strict=True, gt=0)]


class ObservationPolicy(StrictModel):
    sources: Sources
    schedules: Schedules
    backfill: Backfill


class SDOImages(StrictModel):
    enabled: bool = False
    live_seconds: int = Field(default=300, ge=60)
    warmup_seconds: int = Field(default=60, ge=10)
    cleanup_seconds: int = Field(default=3600, ge=60, le=3600)
    warmup_hours_per_batch: int = Field(default=6, ge=1, le=24)


class ClioObservations(StrictModel):
    sdo_images: SDOImages = Field(default_factory=SDOImages)
    observations: dict[str, ObservationPolicy]

    @model_validator(mode='after')
    def compatible_products(self):
        for metric, policy in self.observations.items():
            if metric not in OBSERVATIONS:
                raise ValueError(f'{metric}: unknown observation; supported: {", ".join(OBSERVATIONS)}')
            if OBSERVATIONS[metric].kind == 'file' and policy.backfill.days > 60:
                raise ValueError(f'{metric}: file backfill is bounded to 60 days')
            for mode in ('live', 'historical'):
                sources = getattr(policy.sources, mode)
                if not sources:
                    raise ValueError(f'{metric}: {mode} sources must not be empty')
                for name in sources:
                    product = PRODUCTS.get(name)
                    if (product is None or metric not in product.metrics or mode not in product.modes
                            or product.kind != OBSERVATIONS[metric].kind):
                        raise ValueError(f'{metric}: incompatible or unknown {mode} product {name}')
        return self


def load_observation_config() -> ClioObservations:
    from common.config import get_config
    return ClioObservations.model_validate(get_config().project_config.get('clio', {'observations': {}}))
