"""Read the density artifact produced by the application's forecast job."""
import json

from datetime import UTC, datetime

import pandas as pd
from common.config import get_config
from pydantic import ValidationError

from app.schemas.atmospheric_density import DensityCell, DensityForecast, DensityPoint
from app.services.forecast_products import ArtifactNotReadyError


def load_density_forecast() -> DensityForecast:
    config = get_config()
    entry = config.models_registry["models"]["atmospheric_density"]
    path = config.workdir / entry["forecast_path"]
    try:
        frame = pd.read_csv(path)
        required = {"issue_time", "observed_at", "valid_time", "lead_hours", "driver_mode", "background_method", "dtc_method", "history_start", "dtc_observed_at", *DensityCell.model_fields}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError("Incomplete density artifact")
        for column in ("issue_time", "observed_at", "valid_time", "history_start", "dtc_observed_at"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
            if frame[column].isna().any():
                raise ValueError("Missing artifact timestamp")
        for column in ("background_method", "dtc_method", "history_start", "dtc_observed_at"):
            if frame[column].nunique(dropna=False) != 1:
                raise ValueError("Inconsistent driver preparation metadata")
        interpolated_days = {}
        if "background_interpolated_days" in frame:
            if frame.background_interpolated_days.nunique(dropna=False) != 1:
                raise ValueError("Inconsistent interpolation metadata")
            interpolated_days = json.loads(frame.background_interpolated_days.iloc[0])
        gapfilled = frame.background_method.iloc[0] == "trailing_81_daily_values_linear_gapfill"
        if not isinstance(interpolated_days, dict) or gapfilled != bool(interpolated_days):
            raise ValueError("Missing or inconsistent interpolation metadata")
        for name, dates in interpolated_days.items():
            if name not in {"f10_7", "s10", "m10", "y10"} or not isinstance(dates, list) or not 1 <= len(dates) <= 2:
                raise ValueError("Invalid interpolation metadata")
        issue = frame.issue_time.iloc[0]
        observed = frame.observed_at.iloc[0]
        now = pd.Timestamp(datetime.now(UTC))
        if (frame.issue_time.nunique() != 1 or frame.observed_at.nunique() != 1
                or not frame.driver_mode.eq("observed_persistence").all()
                or observed > issue or issue > now
                or frame.history_start.iloc[0] > observed
                or frame.dtc_observed_at.iloc[0] > issue
                or now - issue > pd.Timedelta(hours=entry.get("max_age_hours", 6))):
            raise ValueError("Stale or inconsistent density artifact")
        if not ((frame.valid_time - issue).dt.total_seconds() / 3600).eq(frame.lead_hours).all():
            raise ValueError("Inconsistent lead times")
        if frame.duplicated(["lead_hours", "altitude_km", "latitude_deg"]).any():
            raise ValueError("Duplicate grid cells")
        predictions = [DensityPoint(
            valid_time=group.valid_time.iloc[0], lead_hours=lead,
            cells=[DensityCell(**row) for row in group.to_dict("records")],
        ) for lead, group in frame.groupby("lead_hours", sort=True)]
        if [point.lead_hours for point in predictions] != list(range(49)):
            raise ValueError("Incomplete forecast horizon")
        coordinates = set(zip(frame.altitude_km, frame.latitude_deg))
        if any({(cell.altitude_km, cell.latitude_deg) for cell in point.cells} != coordinates
               for point in predictions):
            raise ValueError("Incomplete grid")
        return DensityForecast(issue_time=issue, observed_at=observed, horizon_hours=48,
                               predictions=predictions,
                               history_start=frame.history_start.iloc[0],
                               dtc_observed_at=frame.dtc_observed_at.iloc[0],
                               background_method=frame.background_method.iloc[0],
                               dtc_method=frame.dtc_method.iloc[0],
                               background_interpolated_days=interpolated_days)
    except (OSError, ValueError, KeyError, ValidationError) as exc:
        raise ArtifactNotReadyError("Atmospheric density forecast is not ready") from exc
