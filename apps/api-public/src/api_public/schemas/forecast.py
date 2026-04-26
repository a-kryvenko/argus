from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class HistoryPoint(BaseModel):
    timestamp: datetime | None = None
    V: float
    BX_GSE: float
    BY_GSM: float
    BZ_GSM: float
    N: float


class ForecastRequest(BaseModel):
    reference_timestamp: datetime | None = None
    history: list[HistoryPoint] = Field(min_length=6, max_length=168)
    surya_embedding: list[float] | None = None

    @field_validator('surya_embedding')
    @classmethod
    def validate_embedding(cls, value: list[float] | None):
        if value is not None and len(value) == 0:
            raise ValueError('surya_embedding must be non-empty when provided')
        return value


class ForecastPoint(BaseModel):
    timestamp: datetime
    bx: float
    by: float
    bz: float
    density: float
    std_bx: float
    std_by: float
    std_bz: float
    std_density: float
    southward_bz_probability: float


class ForecastResponse(BaseModel):
    points: list[ForecastPoint]
    regime_probabilities: dict[str, float]
