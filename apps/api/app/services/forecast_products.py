from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.schemas.forecast import (
    BinaryForecast,
    Forecast,
    ForecastPoint,
    QuantileForecast,
    VariableForecast,
)
from app.schemas.metrics import (
    BinaryMetricsPoint,
    BinaryMetricsSeries,
    ContinuousMetrics,
    ContinuousMetricsPoint,
    ForecastMetrics,
    ReliabilityPoint,
    VariableMetrics,
)
from common.config import get_config


@dataclass(frozen=True)
class Variable:
    name: str
    registry_name: str
    unit: str
    source_prefix: str
    thresholds: tuple[float, ...] = ()
    quantiles: bool = False
    threshold_marker: str = "ge_"


@dataclass(frozen=True)
class Product:
    target: str
    visibility: str
    max_horizon_hours: int
    variables: tuple[Variable, ...]


PRODUCTS = {
    "solar-wind-speed": Product("solar-wind-speed", "public", 96, (
        Variable("v", "solar_wind_speed", "km/s", "v", (450, 500, 600), True),
    )),
    "solar-wind-density": Product("solar-wind-density", "public", 96, (
        Variable("n", "plasma_density", "cm^-3", "n", quantiles=True),
    )),
    "hmf": Product("hmf", "private", 48, (
        Variable("bt", "bt", "nT", "bt", (5, 10, 15)),
        Variable("southward_bz", "southward_bz", "nT", "southward_bz", (5, 10, 15)),
    )),
    "solar-radiation": Product("solar-radiation", "private", 48, tuple(
        Variable(name, name, "index" if name != "f10_7" else "sfu", name, quantiles=True)
        for name in ("f10_7", "s10", "m10", "y10")
    )),
    "geomagnetic-activity": Product("geomagnetic-activity", "public", 48, (
        Variable("kp", "kp", "index", "kp", (4, 5, 6), threshold_marker=""),
        Variable("ap", "ap", "index", "ap", quantiles=True),
    )),
    "dst": Product("dst", "public", 48, (
        Variable("dst", "dst", "nT", "dst", quantiles=True),
    )),
}


class ProductNotFoundError(Exception):
    pass


class ArtifactNotReadyError(Exception):
    pass


def get_product(target: str, visibility: str) -> Product:
    product = PRODUCTS.get(target)
    if product is None or product.visibility != visibility:
        raise ProductNotFoundError(target)
    return product


def _registry(variable: Variable) -> dict:
    return get_config().models_registry["models"][variable.registry_name]


def _forecast_path(variable: Variable) -> Path:
    return get_config().workdir / _registry(variable)["forecast_path"]


def _required_columns(variable: Variable) -> set[str]:
    columns = {"issue_time", "valid_time", "lead_hours"}
    if variable.quantiles:
        columns.update(f"{variable.source_prefix}_q{quantile}" for quantile in (10, 50, 90))
    columns.update(
        f"p_{variable.source_prefix}_{variable.threshold_marker}{threshold:g}"
        for threshold in variable.thresholds
    )
    return columns


def _load_variable_frames(product: Product) -> dict[str, pd.DataFrame]:
    frames = {}
    cache: dict[Path, pd.DataFrame] = {}
    for variable in product.variables:
        path = _forecast_path(variable)
        if not path.is_file():
            continue
        if path not in cache:
            cache[path] = pd.read_csv(path, parse_dates=["issue_time", "valid_time"])
        frame = cache[path]
        if not frame.empty and _required_columns(variable).issubset(frame.columns):
            frames[variable.name] = frame
    return frames


def _number(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _variable_forecast(variable: Variable, row: dict) -> VariableForecast:
    continuous = None
    if variable.quantiles:
        continuous = QuantileForecast(
            q10=row[f"{variable.source_prefix}_q10"],
            q50=row[f"{variable.source_prefix}_q50"],
            q90=row[f"{variable.source_prefix}_q90"],
        )
    binary = []
    for threshold in variable.thresholds:
        probability = _number(
            row[f"p_{variable.source_prefix}_{variable.threshold_marker}{threshold:g}"]
        )
        if probability is not None:
            binary.append(BinaryForecast(threshold=threshold, probability=probability))
    return VariableForecast(unit=variable.unit, continuous=continuous, binary=binary)


def load_forecast(product: Product) -> Forecast:
    frames = _load_variable_frames(product)
    if not frames:
        raise ArtifactNotReadyError(product.target)

    variables = {variable.name: variable for variable in product.variables}
    base_frame = next(iter(frames.values()))
    base_frame = base_frame[
        base_frame["lead_hours"] <= product.max_horizon_hours
    ]
    if base_frame.empty:
        raise ArtifactNotReadyError(product.target)
    issue_time = base_frame.iloc[0]["issue_time"]
    rows_by_variable = {
        name: {int(row["lead_hours"]): row for row in frame.to_dict("records")}
        for name, frame in frames.items()
        if frame.iloc[0]["issue_time"] == issue_time
    }

    predictions = []
    for base_row in base_frame.to_dict("records"):
        lead_hours = int(base_row["lead_hours"])
        values = {}
        for name, rows in rows_by_variable.items():
            row = rows.get(lead_hours)
            if row is not None:
                values[name] = _variable_forecast(variables[name], row)
        predictions.append(ForecastPoint(
            valid_time=base_row["valid_time"],
            lead_hours=lead_hours,
            variables=values,
        ))

    available = [variable.name for variable in product.variables if variable.name in rows_by_variable]
    return Forecast(
        target=product.target,
        issue_time=issue_time,
        horizon_hours=max(point.lead_hours for point in predictions),
        available_variables=available,
        predictions=predictions,
    )


def _parse_reliability(value) -> list[ReliabilityPoint]:
    if pd.isna(value) or not value:
        return []
    return [
        ReliabilityPoint(
            predicted_probability=float(pair.split("_", 1)[0]),
            observed_frequency=float(pair.split("_", 1)[1]),
        )
        for pair in str(value).split(";")
    ]


def _metrics_path(variable: Variable, filename: str) -> Path:
    return get_config().workdir / _registry(variable)["metrics"] / filename


def _continuous_metrics(variable: Variable, max_horizon_hours: int) -> ContinuousMetrics | None:
    if not variable.quantiles:
        return None
    path = _metrics_path(variable, "regression.csv")
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["lead_hours"] <= max_horizon_hours]
    return ContinuousMetrics(
        quantiles=[0.1, 0.5, 0.9],
        by_lead_hour=[
            ContinuousMetricsPoint(
                lead_hours=int(row.pop("lead_hours")),
                values={key: _number(value) for key, value in row.items()},
            )
            for row in frame.to_dict("records")
        ],
    )


def _binary_metrics(variable: Variable, max_horizon_hours: int) -> list[BinaryMetricsSeries]:
    series = []
    for threshold in variable.thresholds:
        path = _metrics_path(variable, f"threshold_{threshold:g}.csv")
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["lead_hours"] <= max_horizon_hours]
        rows = [
            BinaryMetricsPoint(
                lead_hours=row["lead_hours"],
                brier_score=_number(row.get("brier")),
                roc_auc=_number(row.get("roc_auc")),
                average_precision=_number(row.get("avg_precision")),
                threat_score=_number(row.get("threat_score")),
                heidke_skill_score=_number(row.get("heidke")),
                reliability=_parse_reliability(row.get("reliability")),
            )
            for row in frame.to_dict("records")
        ]
        series.append(BinaryMetricsSeries(threshold=threshold, by_lead_hour=rows))
    return series


def load_metrics(product: Product) -> ForecastMetrics:
    variables = {}
    for variable in product.variables:
        continuous = _continuous_metrics(variable, product.max_horizon_hours)
        binary = _binary_metrics(variable, product.max_horizon_hours)
        if continuous is not None or binary:
            variables[variable.name] = VariableMetrics(continuous=continuous, binary=binary)
    if not variables:
        raise ArtifactNotReadyError(product.target)
    return ForecastMetrics(target=product.target, variables=variables)
