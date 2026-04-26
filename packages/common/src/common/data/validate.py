# validate.py
import xarray as xr
from .schema import DatasetConfig


def validate_dataset(ds: xr.Dataset, config: DatasetConfig):
    for var in ["aia", "hmi", "solar_wind"]:
        if var not in ds:
            raise ValueError(f"Missing variable: {var}")

    assert "time" in ds.dims

    assert ds["aia"].dims == ("time", "wavelength", "y", "x")
    assert list(ds["wavelength"].values) == config.aia.wavelengths

    assert ds["hmi"].dims == ("time", "component", "y", "x")
    assert list(ds["component"].values) == config.hmi.components

    assert ds["solar_wind"].dims == ("time", "feature")
    assert list(ds["feature"].values) == config.solar_wind.features