from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.inference.jb2008 import JB2008Drivers, jb2008_density_grid


DEFAULT_ALTITUDES_KM = np.arange(200.0, 801.0, 25.0)
DEFAULT_LATITUDES_DEG = np.arange(-90.0, 91.0, 10.0)
DEFAULT_LONGITUDES_DEG = np.arange(-180.0, 180.0, 45.0)

SOLAR_INDICES = ("f10_7", "s10", "m10", "y10")
class AtmosphericDensityForecastService:
    """Build a grid with observed drivers held constant over the horizon."""

    registry_name = "atmospheric_density"
    requires_models = False

    def __init__(self, models: dict | None = None):
        self.models = models or {}

    @staticmethod
    def _driver_record(row) -> JB2008Drivers:
        return JB2008Drivers(
            f10=float(row.f10_7),
            f10_81c=float(row.f10_7_81mean),
            s10=float(row.s10),
            s10_81c=float(row.s10_81mean),
            m10=float(row.m10),
            m10_81c=float(row.m10_81mean),
            y10=float(row.y10),
            y10_81c=float(row.y10_81mean),
            dtc=float(row.dtc),
        )

    def forecast_grid(
        self,
        drivers: pd.DataFrame,
        altitudes_km=DEFAULT_ALTITUDES_KM,
        latitudes_deg=DEFAULT_LATITUDES_DEG,
        longitudes_deg=DEFAULT_LONGITUDES_DEG,
        progress=None,
    ) -> pd.DataFrame:
        drivers = drivers.copy()
        rows = []

        for completed, driver_row in enumerate(drivers.itertuples(index=False), start=1):
            model_drivers = self._driver_record(driver_row)
            samples = jb2008_density_grid(
                driver_row.valid_time.to_pydatetime(), latitudes_deg,
                longitudes_deg, altitudes_km, model_drivers,
            )
            means = samples.mean(axis=2)
            percentiles = np.quantile(samples, [0.1, 0.9], axis=2)
            for a, altitude in enumerate(altitudes_km):
                for b, latitude in enumerate(latitudes_deg):
                    rows.append({
                        "valid_time": driver_row.valid_time,
                        "observed_at": getattr(driver_row, "observed_at", None),
                        "driver_mode": "observed_persistence",
                        "background_method": driver_row.background_method,
                        "background_interpolated_days": driver_row.background_interpolated_days,
                        "dtc_method": driver_row.dtc_method,
                        "history_start": driver_row.history_start,
                        "dtc_observed_at": driver_row.dtc_observed_at,
                        "altitude_km": float(altitude),
                        "latitude_deg": float(latitude),
                        "rho_kg_m3": float(means[a, b]),
                        "rho_lon_p10_kg_m3": float(percentiles[0, a, b]),
                        "rho_lon_p90_kg_m3": float(percentiles[1, a, b]),
                        "f10_7": model_drivers.f10,
                        "s10": model_drivers.s10,
                        "m10": model_drivers.m10,
                        "y10": model_drivers.y10,
                        "dtc": model_drivers.dtc,
                        "f10_7_81mean": model_drivers.f10_81c,
                        "s10_81mean": model_drivers.s10_81c,
                        "m10_81mean": model_drivers.m10_81c,
                        "y10_81mean": model_drivers.y10_81c,
                    })

            if progress is not None:
                progress(completed, len(drivers))

        result = pd.DataFrame(rows)
        if result.empty or not np.isfinite(result["rho_kg_m3"]).all():
            raise RuntimeError("JB2008 produced an empty or non-finite density forecast")
        return result
