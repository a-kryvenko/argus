from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.config import get_config
from forecast_core.models.jb2008 import JB2008Drivers, jb2008_density


DEFAULT_ALTITUDES_KM = np.arange(200.0, 801.0, 25.0)
DEFAULT_LATITUDES_DEG = np.arange(-90.0, 91.0, 10.0)
DEFAULT_LONGITUDES_DEG = np.arange(-180.0, 180.0, 45.0)

SOLAR_INDICES = ("f10_7", "s10", "m10", "y10")
class AtmosphericDensityForecastService:
    """Build a gridded JB2008 density product from forecast driver CSV files.

    Solar forecast timestamps are the effective JB2008 input times: producer
    services must apply the model's 1/1/2/5-day lags for F10.7/S10/M10/Y10.
    """

    registry_name = "atmospheric_density"
    requires_models = False

    def __init__(self, models: dict | None = None):
        self.models = models or {}

    @staticmethod
    def _registry_path(name: str) -> Path:
        config = get_config()
        entry = config.models_registry["models"].get(name)
        if entry is None or not entry.get("forecast_path"):
            raise KeyError(f"models_registry has no forecast_path for {name}")
        return config.workdir / entry["forecast_path"]

    @staticmethod
    def _value_column(frame: pd.DataFrame, name: str) -> str:
        candidates = (f"{name}_q50", "q50", name)
        column = next((candidate for candidate in candidates if candidate in frame), None)
        if column is None:
            raise ValueError(f"{name} forecast must contain one of {candidates}")
        return column

    @staticmethod
    def _background_column(frame: pd.DataFrame, name: str) -> str:
        candidates = (f"{name}_81c_q50", f"{name}_81c")
        column = next((candidate for candidate in candidates if candidate in frame), None)
        if column is None:
            raise ValueError(f"{name} forecast must contain one of {candidates}")
        return column

    @classmethod
    def _read_forecast(cls, name: str) -> pd.DataFrame:
        path = cls._registry_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} forecast: {path}")
        frame = pd.read_csv(path, parse_dates=["valid_time"])
        frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True).dt.floor("h")
        value_column = cls._value_column(frame, name)
        columns = ["valid_time", value_column]
        rename = {value_column: name}
        if name in SOLAR_INDICES:
            background_column = cls._background_column(frame, name)
            columns.append(background_column)
            rename[background_column] = f"{name}_81c"
        return (
            frame[columns]
            .rename(columns=rename)
            .dropna()
            .drop_duplicates("valid_time", keep="last")
            .sort_values("valid_time")
        )

    @classmethod
    def load_drivers(cls) -> pd.DataFrame:
        """Load synchronized drivers; DTC is mandatory and never inferred from Dst."""
        frames = [cls._read_forecast(name) for name in (*SOLAR_INDICES, "dtc")]
        drivers = frames[0]
        for frame in frames[1:]:
            drivers = drivers.merge(frame, on="valid_time", how="inner")

        if drivers.empty:
            raise ValueError("Driver forecasts have no common valid_time values")

        return drivers

    @staticmethod
    def _driver_record(row) -> JB2008Drivers:
        return JB2008Drivers(
            f10=float(row.f10_7),
            f10_81c=float(row.f10_7_81c),
            s10=float(row.s10),
            s10_81c=float(row.s10_81c),
            m10=float(row.m10),
            m10_81c=float(row.m10_81c),
            y10=float(row.y10),
            y10_81c=float(row.y10_81c),
            dtc=float(row.dtc),
        )

    def forecast_grid(
        self,
        drivers: pd.DataFrame | None = None,
        altitudes_km=DEFAULT_ALTITUDES_KM,
        latitudes_deg=DEFAULT_LATITUDES_DEG,
        longitudes_deg=DEFAULT_LONGITUDES_DEG,
    ) -> pd.DataFrame:
        drivers = self.load_drivers() if drivers is None else drivers.copy()
        rows = []

        for driver_row in drivers.itertuples(index=False):
            model_drivers = self._driver_record(driver_row)
            for altitude in altitudes_km:
                for latitude in latitudes_deg:
                    samples = [
                        jb2008_density(
                            driver_row.valid_time.to_pydatetime(),
                            float(latitude),
                            float(longitude),
                            float(altitude),
                            model_drivers,
                        )
                        for longitude in longitudes_deg
                    ]
                    rows.append({
                        "valid_time": driver_row.valid_time,
                        "altitude_km": float(altitude),
                        "latitude_deg": float(latitude),
                        "rho_kg_m3": float(np.mean(samples)),
                        "rho_lon_p10_kg_m3": float(np.quantile(samples, 0.10)),
                        "rho_lon_p90_kg_m3": float(np.quantile(samples, 0.90)),
                        "f10_7": model_drivers.f10,
                        "s10": model_drivers.s10,
                        "m10": model_drivers.m10,
                        "y10": model_drivers.y10,
                        "dtc": model_drivers.dtc,
                    })

        result = pd.DataFrame(rows)
        if result.empty or not np.isfinite(result["rho_kg_m3"]).all():
            raise RuntimeError("JB2008 produced an empty or non-finite density forecast")
        return result
