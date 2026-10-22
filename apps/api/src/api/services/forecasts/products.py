from dataclasses import dataclass

import pandas as pd
from api.schemas.forecast import (
    BinaryForecast,
    Forecast,
    ForecastPoint,
    ForecastMetadata,
    ForecastVariableMetadata,
    QuantileForecast,
    VariableForecast,
)
from api.schemas.metrics import ForecastMetrics
from common.config import get_config
from common.density_contract import parse_density_frame
from common.schemas.atmospheric_density import DensityForecast
from api.services.errors import ArtifactNotReadyError
from api.services.forecasts.client import read_frames
from common.schemas.solar_wind_events import CMESpeedOutlook


@dataclass(frozen=True)
class Variable:
    name: str
    registry_name: str
    unit: str
    source_prefix: str
    thresholds: tuple[float, ...] = ()
    quantiles: bool = False


@dataclass(frozen=True)
class Product:
    target: str
    max_horizon_hours: int
    variables: tuple[Variable, ...]


PRODUCTS = {
    "solar-wind-speed": Product("solar-wind-speed", 96, (
        Variable("v", "plasma_speed_quantile", "km/s", "v", quantiles=True),
        Variable("v", "plasma_speed_threshold", "km/s", "v", (450, 500, 600)),
    )),
    "solar-wind-density": Product("solar-wind-density", 96, (
        Variable("n", "plasma_density_quantile", "cm^-3", "n", quantiles=True),
    )),
    "solar-wind-temperature": Product("solar-wind-temperature", 96, (
        Variable("t", "plasma_temperature_quantile", "K", "t", quantiles=True),
    )),
    "hmf": Product("hmf", 48, (
        Variable("bt", "hmf_total_threshold", "nT", "bt", (5, 10, 15)),
        Variable("bs", "hmf_southward_threshold", "nT", "bs", (5, 10, 15)),
    )),
    "solar-radiation": Product("solar-radiation", 48, (
        Variable("f10_7", "f10_7_quantile", "sfu", "f107", quantiles=True),
        Variable("s10", "s10_quantile", "index", "s10", quantiles=True),
        Variable("m10", "m10_quantile", "index", "m10", quantiles=True),
        Variable("y10", "y10_quantile", "index", "y10", quantiles=True),
    )),
    "geomagnetic-activity": Product("geomagnetic-activity", 48, (
        Variable("kp", "kp_threshold", "index", "kp", (4, 5, 6)),
        Variable("ap", "ap_quantile", "index", "ap", quantiles=True),
    )),
    "dst": Product("dst", 48, (
        Variable("dst", "dst_quantile", "nT", "dst", quantiles=True),
    )),
}


def _required_columns(variable: Variable) -> set[str]:
    columns = {"issue_time", "valid_time", "lead_hours"}
    if variable.quantiles:
        columns.update(f"{variable.source_prefix}_q{quantile}" for quantile in (10, 50, 90))
    columns.update(
        f"p_{variable.source_prefix}_ge_{threshold:g}"
        for threshold in variable.thresholds
    )
    return columns


def _load_variable_frames(product: Product) -> list[tuple[Variable, pd.DataFrame]]:
    sources = read_frames(product.target)
    frames = []
    for variable in product.variables:
        frame = sources.get(variable.registry_name)
        if frame is None or frame.empty or not _required_columns(variable).issubset(frame.columns):
            raise ArtifactNotReadyError(product.target)
        frames.append((variable, frame))
    return frames


def _variable_forecast(variable: Variable, row: dict) -> VariableForecast:
    continuous = None
    speed = variable.name == "v" and variable.unit == "km/s"
    value = round if speed else float
    if variable.name == "n" and variable.unit == "cm^-3":
        value = lambda number: round(float(number), 3)
    if variable.quantiles:
        continuous = QuantileForecast(
            q10=value(row[f"{variable.source_prefix}_q10"]),
            q50=value(row[f"{variable.source_prefix}_q50"]),
            q90=value(row[f"{variable.source_prefix}_q90"]),
        )
    binary = []
    for threshold in variable.thresholds:
        probability = row[f"p_{variable.source_prefix}_ge_{threshold:g}"]
        if not pd.isna(probability):
            binary.append(BinaryForecast(threshold=int(threshold) if speed else threshold, probability=float(probability)))
    return VariableForecast(continuous=continuous, binary=binary)


def product_metadata(product: Product, available: list[str]) -> ForecastMetadata:
    return ForecastMetadata(variables={
        variable.name: ForecastVariableMetadata(unit=variable.unit)
        for variable in product.variables if variable.name in available
    })


def load_density_forecast() -> DensityForecast:
    config = get_config()
    entry = config.models_registry["models"]["atmospheric_density"]
    try:
        frame = read_frames("atmospheric-density")["atmospheric_density"]
        return parse_density_frame(frame, max_age_hours=entry.get("max_age_hours", 6))
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactNotReadyError("Atmospheric density forecast is not ready") from exc


def load_forecast(product: Product, *, meta: bool = False) -> Forecast:
    frames = _load_variable_frames(product)
    if not frames:
        raise ArtifactNotReadyError(product.target)

    issue_time = frames[0][1].iloc[0]['issue_time']
    # Products may contain variables with different supported horizons.
    base_frame = pd.concat([frame for _, frame in frames
                            if frame.iloc[0]['issue_time'] == issue_time], ignore_index=True)
    base_frame = base_frame.drop_duplicates('lead_hours').sort_values('lead_hours')
    base_frame = base_frame[
        base_frame["lead_hours"] <= product.max_horizon_hours
    ]
    if base_frame.empty:
        raise ArtifactNotReadyError(product.target)
    issue_time = base_frame.iloc[0]["issue_time"]
    rows_by_source = [
        (
            variable,
            {int(row["lead_hours"]): row for row in frame.to_dict("records")},
        )
        for variable, frame in frames
        if frame.iloc[0]["issue_time"] == issue_time
    ]

    predictions = []
    for base_row in base_frame.to_dict("records"):
        lead_hours = int(base_row["lead_hours"])
        values = {}
        for variable, rows in rows_by_source:
            row = rows.get(lead_hours)
            if row is not None:
                addition = _variable_forecast(variable, row)
                current = values.setdefault(variable.name, VariableForecast())
                current.continuous = current.continuous or addition.continuous
                current.binary.extend(addition.binary)
        predictions.append(ForecastPoint(
            valid_time=base_row["valid_time"],
            lead_hours=lead_hours,
            variables=values,
        ))

    available = list(dict.fromkeys(variable.name for variable, _ in rows_by_source))
    cme_speed = None
    if product.target == 'solar-wind-speed':
        for variable, frame in frames:
            if variable.registry_name == 'plasma_speed_quantile' and 'cme_speed_json' in frame:
                payloads = frame.cme_speed_json.dropna()
                payloads = payloads[payloads.astype(str).str.len() > 0]
                if len(payloads) == 1:
                    cme_speed = CMESpeedOutlook.model_validate_json(payloads.iloc[0])
                    times = {point.valid_time for point in predictions}
                    cme_speed.points = [point for point in cme_speed.points if point.valid_time in times]
    return Forecast(
        target=product.target,
        issue_time=issue_time,
        horizon_hours=max(point.lead_hours for point in predictions),
        available_variables=available,
        predictions=predictions,
        meta=product_metadata(product, available) if meta else None,
        cme_speed=cme_speed,
    )


def load_metrics(product: Product, *, meta: bool = False) -> ForecastMetrics:
    config = get_config()
    for variable in product.variables:
        report_path = config.models_registry["models"][variable.registry_name].get("metrics_report")
        if report_path:
            path = config.workdir / report_path
            if not path.is_file():
                raise ArtifactNotReadyError(product.target)
            report = ForecastMetrics.model_validate_json(path.read_text())
            if report.target != product.target:
                raise ValueError("Evaluation report target mismatch")
            for metrics in report.variables.values():
                if metrics.continuous:
                    metrics.continuous.by_lead_hour = [r for r in metrics.continuous.by_lead_hour
                                                       if r.lead_hours <= product.max_horizon_hours]
                for series in metrics.binary:
                    series.by_lead_hour = [r for r in series.by_lead_hour if r.lead_hours <= product.max_horizon_hours]
            report.meta = product_metadata(product, list(report.variables)) if meta else None
            return report
    raise ArtifactNotReadyError(product.target)
