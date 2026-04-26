# builder.py
import xarray as xr
import numpy as np
from .schema import DatasetConfig

def build_dataset(
    config: DatasetConfig,
    time: np.ndarray,
    aia_data: np.ndarray,   # shape: (time, wavelength, y, x)
    hmi_data: np.ndarray,   # shape: (time, component, y, x)
    sw_data: np.ndarray     # shape: (time, feature)
) -> xr.Dataset:

    ds = xr.Dataset(
        {
            "aia": (("time", "wavelength", "y", "x"), aia_data),
            "hmi": (("time", "component", "y", "x"), hmi_data),
            "solar_wind": (("time", "feature"), sw_data),
        },
        coords={
            "time": time,
            "wavelength": config.aia.wavelengths,
            "component": config.hmi.components,
            "feature": config.solar_wind.features,
        },
        attrs={
            "schema_version": config.schema_version
        }
    )

    return ds