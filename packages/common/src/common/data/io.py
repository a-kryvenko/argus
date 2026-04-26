import xarray as xr
from pathlib import Path


def save_image_snapshot(ds: xr.Dataset, root: Path, timestamp: str):
    path = root / f"{timestamp}.nc"
    ds.to_netcdf(path)

def load_image_snapshot(root: Path, timestamp: str) -> xr.Dataset:
    path = root / f"{timestamp}.nc"
    return xr.open_dataset(path)

def save_solar_wind(ds: xr.Dataset, path: str):
    ds.to_netcdf(path)

def load_solar_wind(path: str) -> xr.Dataset:
    return xr.open_dataset(path)
