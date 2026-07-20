from typing import Any, Literal

from pydantic import BaseModel, Field


class ReliabilityPoint(BaseModel):
    predicted_probability: float
    observed_frequency: float


class BinaryMetricsPoint(BaseModel):
    lead_hours: int
    brier_score: float | None = None
    roc_auc: float | None = None
    average_precision: float | None = None
    threat_score: float | None = None
    heidke_skill_score: float | None = None
    reliability: list[ReliabilityPoint] = Field(default_factory=list)


class BinaryMetricsSeries(BaseModel):
    threshold: float
    operator: Literal["gte"] = "gte"
    by_lead_hour: list[BinaryMetricsPoint]


class ContinuousMetricsPoint(BaseModel):
    lead_hours: int
    values: dict[str, Any]


class ContinuousMetrics(BaseModel):
    quantiles: list[float]
    by_lead_hour: list[ContinuousMetricsPoint]


class VariableMetrics(BaseModel):
    continuous: ContinuousMetrics | None = None
    binary: list[BinaryMetricsSeries] = Field(default_factory=list)


class ForecastMetrics(BaseModel):
    target: str
    variables: dict[str, VariableMetrics]
