"""Validated Clio sources, collection horizons and retention."""
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clio.ingestion.utils.products import OBSERVATIONS, PRODUCTS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Sources(StrictModel):
    live: list[str] = Field(default_factory=list)
    historical: list[str] = Field(default_factory=list)

    @field_validator('live', 'historical')
    @classmethod
    def unique_products(cls, values):
        if len(values) != len(set(values)):
            raise ValueError('Source priorities must not contain duplicates')
        return values


class ObservationPolicy(StrictModel):
    sources: Sources
    backfill_days: int = Field(strict=True, ge=0)
    retention_days: int | None = Field(default=None, strict=True, gt=0)
    resolution_minutes: int = Field(strict=True, gt=0)
    live_resolution_minutes: int = Field(strict=True, gt=0)
    poll_minutes: int = Field(strict=True, gt=0)
    freshness_minutes: int = Field(strict=True, gt=0)

    @model_validator(mode='after')
    def consistent_windows(self):
        if not self.sources.live and not self.sources.historical:
            raise ValueError('Configure at least one source')
        if self.sources.historical and self.backfill_days == 0:
            raise ValueError('Historical sources require positive backfill_days')
        if not self.sources.historical and self.backfill_days:
            raise ValueError('backfill_days requires historical sources')
        if self.retention_days is not None and self.backfill_days > self.retention_days:
            raise ValueError('backfill exceeds retention')
        return self


class SolarWindEvents(StrictModel):
    poll_minutes: int = Field(default=15, strict=True, gt=0)
    enabled: bool = False
    lookback_days: int = Field(default=30, strict=True, ge=7, le=90)
    retention_days: int = Field(default=30, strict=True, ge=1, le=90)


class Retention(StrictModel):
    live_days: int = Field(default=6, strict=True, gt=0)
    normalized_days: int = Field(default=60, strict=True, gt=0)


class ClioObservations(StrictModel):
    retention: Retention = Field(default_factory=Retention)
    solar_wind_events: SolarWindEvents = Field(default_factory=SolarWindEvents)
    observations: dict[str, ObservationPolicy]

    @model_validator(mode='after')
    def compatible_products(self):
        for metric, policy in self.observations.items():
            if metric not in OBSERVATIONS:
                raise ValueError(f'{metric}: unknown observation; supported: {", ".join(OBSERVATIONS)}')
            definition = OBSERVATIONS[metric]
            if definition.kind == 'file' and policy.retention_days is None:
                raise ValueError(f'{metric}: retention_days is required')
            for mode in ('live', 'historical'):
                sources = getattr(policy.sources, mode)
                for name in sources:
                    product = PRODUCTS.get(name)
                    if (product is None or metric not in product.metrics or mode not in product.modes
                            or product.kind != OBSERVATIONS[metric].kind):
                        raise ValueError(f'{metric}: incompatible or unknown {mode} product {name}')
                    resolution = policy.live_resolution_minutes if mode == 'live' else policy.resolution_minutes
                    if product.slot_minutes and resolution % product.slot_minutes:
                        raise ValueError(f'{metric}: {mode} resolution must be a multiple of {product.slot_minutes} minutes for {name}')
        return self


def load_observation_config() -> ClioObservations:
    from common.config import get_config
    return ClioObservations.model_validate(get_config().project_config.get('clio', {'observations': {}}))
