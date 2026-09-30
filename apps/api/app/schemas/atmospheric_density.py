"""Public density projection; internal scientific validation stays in common."""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field
from common.schemas.atmospheric_density import DensityForecast as InternalDensityForecast, DensityPoint
from app.schemas.metadata import metadata_field


class DensityMetadata(BaseModel):
    model: Literal['JB2008'] = 'JB2008'
    history_start: datetime
    background_method: Literal['trailing_81_daily_values', 'trailing_81_daily_values_linear_gapfill']
    background_interpolated_days: dict[str, list[date]]
    dtc_method: Literal['causal_dst_ap_v1']


class DensityForecast(BaseModel):
    target: Literal['atmospheric-density'] = 'atmospheric-density'
    driver_mode: Literal['observed_persistence'] = 'observed_persistence'
    issue_time: datetime
    observed_at: datetime
    dtc_observed_at: datetime
    background_interpolated: bool = Field(description='True when background drivers include gap-filled daily values. Details are available with meta=true.')
    horizon_hours: int
    predictions: list[DensityPoint]
    meta: DensityMetadata | None = metadata_field()

    @classmethod
    def from_internal(cls, forecast: InternalDensityForecast, *, meta: bool = False):
        data = forecast.model_dump()
        return cls(**data, background_interpolated=any(forecast.background_interpolated_days.values()),
                   meta=DensityMetadata.model_validate(data) if meta else None)
