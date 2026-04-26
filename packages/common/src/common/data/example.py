# main.py
import numpy as np
from .schema import DatasetConfig, AIAConfig, HMIConfig, SolarWindConfig
from .builder import build_dataset
from .io import load_image_snapshot, load_solar_wind
from .validate import validate_dataset

config = DatasetConfig(
    aia=AIAConfig(wavelengths=[
        "aia94","aia131","aia171","aia193",
        "aia211","aia304","aia335","aia1600"
    ]),
    hmi=HMIConfig(components=[
        "hmi_m","hmi_bx","hmi_by","hmi_bz","hmi_v"
    ]),
    solar_wind=SolarWindConfig(features=[
        "Bx","By","Bz","V","N","T"
    ]),
    image_shape=(4096, 4096)
)

def load_full_sample(image_root, sw_path, timestamp):
    img = load_image_snapshot(image_root, timestamp)
    sw = load_solar_wind(sw_path)

    # align solar wind to this timestamp
    sw_t = sw.sel(time=img.time, method="nearest")

    return img, sw_t

T = 10
time = np.arange(T)

aia = np.random.rand(T, 8, 4096, 4096).astype("float32")
hmi = np.random.rand(T, 5, 4096, 4096).astype("float32")
sw  = np.random.rand(T, 6).astype("float32")

ds = build_dataset(config, time, aia, hmi, sw)

validate_dataset(ds, config)

save_dataset(ds, "solar_dataset.nc")

ds2 = load_dataset("solar_dataset.nc")
validate_dataset(ds2, config)