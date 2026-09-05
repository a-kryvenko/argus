from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

LEAD_HOURS_DESCRIPTION = (
    "Number of hours between forecast issuance (issue_time) and the time "
    "being predicted (valid_time)."
)
VARIABLES_DESCRIPTION = (
    "Values keyed by variable name. See the endpoint's product table for the "
    "variables and units supported by each target. Only available variables "
    "are included; individual forecast points may contain a subset."
)


class QuantileForecast(BaseModel):
    """Quantile forecast in the parent variable's units, not probabilities."""

    q10: float = Field(description="Predicted 10th percentile (quantile 0.1).", examples=[380.0])
    q50: float = Field(description="Predicted median (quantile 0.5).", examples=[420.0])
    q90: float = Field(description="Predicted 90th percentile (quantile 0.9).", examples=[480.0])


class BinaryForecast(BaseModel):
    """Probability that the variable is greater than or equal to a threshold."""

    threshold: float = Field(description="Inclusive threshold in the parent variable's units.", examples=[450.0])
    probability: float = Field(description="Probability of value >= threshold, from 0 to 1 (not percent).", examples=[0.25])


class VariableForecast(BaseModel):
    """Available quantile and threshold forecasts for one physical variable."""

    unit: str = Field(description="Unit shared by quantile values and binary thresholds; index denotes an index value.", examples=["km/s", "nT", "index"])
    continuous: QuantileForecast | None = Field(default=None, description="Quantile forecast, or null when unavailable or unsupported for this variable.")
    binary: list[BinaryForecast] = Field(default_factory=list, description="Available threshold probabilities; empty when unavailable or unsupported. Thresholds with missing probabilities are omitted.")


class ForecastPoint(BaseModel):
    """Forecast values for a single predicted time."""

    valid_time: datetime = Field(description="Date and time for which the forecast predicts values (ISO 8601).", examples=["2026-09-05T01:00:00Z"])
    lead_hours: int = Field(description=LEAD_HOURS_DESCRIPTION, examples=[1])
    variables: dict[str, VariableForecast] = Field(description=VARIABLES_DESCRIPTION)


class Forecast(BaseModel):
    """Forecast for one product, issued at a single time."""

    target: str = Field(description="Forecast product identifier; see the endpoint's visibility/target table.", examples=["solar-wind-speed"])
    issue_time: datetime = Field(description="Date and time when the forecast was made (ISO 8601).", examples=["2026-09-05T00:00:00Z"])
    horizon_hours: int = Field(description="Largest lead_hours among returned predictions; may be shorter than the product's maximum horizon.", examples=[1])
    available_variables: list[str] = Field(description="Variable names available for this forecast issuance. A particular prediction may contain only a subset.", examples=[["v"]])
    predictions: list[ForecastPoint] = Field(description="Forecast points, each with its own valid_time and lead_hours.")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "target": "solar-wind-speed",
        "issue_time": "2026-09-05T00:00:00Z",
        "horizon_hours": 1,
        "available_variables": ["v"],
        "predictions": [{
            "valid_time": "2026-09-05T01:00:00Z", "lead_hours": 1,
            "variables": {"v": {"unit": "km/s", "continuous": {
                "q10": 380.0, "q50": 420.0, "q90": 480.0,
            }, "binary": [
                {"threshold": 450.0, "probability": 0.25},
                {"threshold": 500.0, "probability": 0.05},
                {"threshold": 600.0, "probability": 0.01},
            ]}},
        }],
    }]})
