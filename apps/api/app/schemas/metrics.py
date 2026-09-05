from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.forecast import LEAD_HOURS_DESCRIPTION


class ReliabilityPoint(BaseModel):
    """A calibration point comparing predicted probability with observed frequency."""

    predicted_probability: float = Field(description="Predicted event probability for this calibration point, from 0 to 1.", examples=[0.25])
    observed_frequency: float = Field(description="Observed fraction of events for this calibration point, from 0 to 1.", examples=[0.23])


class BinaryMetricsPoint(BaseModel):
    """Threshold forecast evaluation at one lead time. Null scores are unavailable."""

    lead_hours: int = Field(description=LEAD_HOURS_DESCRIPTION, examples=[1])
    brier_score: float | None = Field(default=None, description="Mean squared error of event probabilities; lower is better. Null when unavailable.", examples=[0.12])
    roc_auc: float | None = Field(default=None, description="Area under the ROC curve; higher is better. Null when unavailable.", examples=[0.85])
    average_precision: float | None = Field(default=None, description="Average precision summarizing precision and recall; higher is better. Null when unavailable.", examples=[0.72])
    threat_score: float | None = Field(default=None, description="Hits divided by hits + misses + false alarms (critical success index). Null when unavailable.", examples=[0.6])
    heidke_skill_score: float | None = Field(default=None, description="Classification skill relative to chance agreement; 1 is perfect, 0 means no skill over chance. Null when unavailable.", examples=[0.55])
    reliability: list[ReliabilityPoint] = Field(default_factory=list, description="Calibration points comparing predicted probabilities with observed frequencies; empty when unavailable.")


class BinaryMetricsSeries(BaseModel):
    """Evaluation of an inclusive threshold event across lead times."""

    threshold: float = Field(description="Event threshold in the variable's units; see the endpoint's product table.", examples=[450.0])
    operator: Literal["gte"] = Field(default="gte", description="gte means the event is variable value >= threshold.")
    by_lead_hour: list[BinaryMetricsPoint] = Field(description="Threshold metrics grouped by forecast lead time in hours.")


class ContinuousMetricsPoint(BaseModel):
    """Quantile/regression forecast evaluation at one lead time."""

    lead_hours: int = Field(description=LEAD_HOURS_DESCRIPTION, examples=[1])
    values: dict[str, Any] = Field(description="Metric names mapped to numeric scores, or null for missing scores. Available names depend on the product's evaluation output.")


class ContinuousMetrics(BaseModel):
    """Evaluation of the quantile forecast across lead times."""

    quantiles: list[float] = Field(description="Evaluated quantile levels as fractions, corresponding to q10, q50 and q90 in the forecast.", examples=[[0.1, 0.5, 0.9]])
    by_lead_hour: list[ContinuousMetricsPoint] = Field(description="Quantile/regression metrics grouped by forecast lead time in hours.")


class VariableMetrics(BaseModel):
    """Available evaluation results for one forecast variable."""

    continuous: ContinuousMetrics | None = Field(default=None, description="Quantile/regression metrics, or null when unavailable or unsupported.")
    binary: list[BinaryMetricsSeries] = Field(default_factory=list, description="Metrics for available thresholds; empty when unavailable or unsupported.")


class ForecastMetrics(BaseModel):
    """Evaluation metrics for one product; these are scores, not forecast values."""

    target: str = Field(description="Forecast product identifier; see the endpoint's visibility/target table.", examples=["solar-wind-speed"])
    variables: dict[str, VariableMetrics] = Field(description="Metrics keyed by variable name. See the endpoint's product table for names and units. Variables without available metrics are omitted.")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "target": "solar-wind-speed",
        "variables": {"v": {"continuous": None, "binary": [{
            "threshold": 450.0, "operator": "gte", "by_lead_hour": [{
                "lead_hours": 1, "brier_score": 0.12, "roc_auc": 0.85,
                "average_precision": 0.72, "threat_score": 0.6,
                "heidke_skill_score": 0.55,
                "reliability": [{"predicted_probability": 0.25, "observed_frequency": 0.23}],
            }],
        }]}},
    }]})
