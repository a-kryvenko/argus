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


class ForecastInputs(BaseModel):
    schema_version: Literal[1] = 1
    as_of: AwareDatetime
    read_at: AwareDatetime
    observations: Observation
    measurements: list[SourceMeasurement] = Field(default_factory=list)
