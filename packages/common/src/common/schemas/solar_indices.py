"""Calibrated solar-index observations recorded by Prophet."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, FiniteFloat, model_validator


class SolarIndexMeasurement(BaseModel):
    metric: Literal['f10_7', 's10', 'm10', 'y10']
    observed_at: AwareDatetime
    value: FiniteFloat


class SolarIndices(BaseModel):
    contract_version: Literal[1] = 1
    as_of: AwareDatetime
    items: list[SolarIndexMeasurement]

    @model_validator(mode='after')
    def causal(self):
        if any(point.observed_at > self.as_of for point in self.items):
            raise ValueError('Solar observations must precede the snapshot cutoff')
        return self
