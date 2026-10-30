"""Validated Clio sources, collection horizons and retention."""
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


class ObservationPolicy(StrictModel):
    sources: Sources
    backfill_days: int = Field(strict=True, gt=0)


class SDOImages(StrictModel):
    enabled: bool = False
    backfill_days: int = Field(default=45, strict=True, ge=1, le=45)


class SolarWindEvents(StrictModel):
    enabled: bool = False
    lookback_days: int = Field(default=30, strict=True, ge=7, le=90)


class Retention(StrictModel):
    live_days: int = Field(default=6, strict=True, gt=0)
    normalized_days: int = Field(default=60, strict=True, gt=0)


class ClioObservations(StrictModel):
    retention: Retention = Field(default_factory=Retention)
    solar_wind_events: SolarWindEvents = Field(default_factory=SolarWindEvents)
    sdo_images: SDOImages = Field(default_factory=SDOImages)
    sdo_browse_enabled: bool = False
    observations: dict[str, ObservationPolicy]

    @model_validator(mode='after')
    def compatible_products(self):
        for metric, policy in self.observations.items():
            if metric not in OBSERVATIONS:
                raise ValueError(f'{metric}: unknown observation; supported: {", ".join(OBSERVATIONS)}')
            limit = 90 if metric == 'goes' else 60
            if OBSERVATIONS[metric].kind == 'file' and policy.backfill_days > limit:
                raise ValueError(f'{metric}: file backfill is bounded to {limit} days')
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
