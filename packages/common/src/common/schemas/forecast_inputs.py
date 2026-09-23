"""Version 1 observation read contract, independent of storage and model code."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, FiniteFloat
from common.schemas.observation import Observation

DensityMetric = Literal['f10_7', 's10', 'm10', 'y10', 'dst', 'ap']
DENSITY_METRICS = ('f10_7', 's10', 'm10', 'y10', 'dst', 'ap')


class SourceMeasurement(BaseModel):
    metric: DensityMetric
    value: FiniteFloat
    observed_at: AwareDatetime


class SpeedObservation(BaseModel):
    """Observed hourly speed, without interpolation or gap filling."""
    issue_time: AwareDatetime
    v: FiniteFloat


class AIAFeatureFrame(BaseModel):
    """Six-hour model input derived from the hourly owner archive as of the read."""
    slot_at: AwareDatetime
    observed_at: AwareDatetime
    available_at: AwareDatetime
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    features: dict[str, FiniteFloat | None]


class ForecastInputs(BaseModel):
    schema_version: Literal[1] = 1
    as_of: AwareDatetime
    read_at: AwareDatetime
    observations: Observation
    measurements: list[SourceMeasurement] = Field(default_factory=list)
    speed_observations: list[SpeedObservation] = Field(default_factory=list)

    aia_frames: list[AIAFeatureFrame] = Field(default_factory=list)
