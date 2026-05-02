import requests
from datetime import datetime
import pandas as pd
from pathlib import Path
from common.schema_raw import Image, SensorData, AIAImages, HMIImages, RawObservationDataPoint
from common.config import get_config
from forecast_core.data_pipelines.transformer import build_bz

SDO_AIA_LIVE_URL = "https://jsoc1.stanford.edu/data/aia/images/image_times"
AIA_WAVELENGTHS = [
    "94",
    "131",
    "171",
    "193",
    "211",
    "304",
    "335",
    "1600"
]

def _fetch_aia(data_root: Path) -> AIAImages:
    output_dir = data_root / "live/raw/aia"
    output_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(SDO_AIA_LIVE_URL)

    if r.status_code != 200:
        raise Exception("SDO server connection error: " + r.status_code)
    
    images = dict()
    for wavelength in AIA_WAVELENGTHS:
        images[wavelength] = ""
    
    for line in r.text.split("\n"):
        parts = line.split()
        if len(parts) == 2:
            try:
                parts[0] = parts[0]
                if parts[0] in images:
                    response = requests.get(parts[1])
                    response.raise_for_status()
                    image_name = parts[0] + ".jp2"
                    with open(output_dir / image_name, "wb") as f:
                        f.write(response.content)
                    images[parts[0]] = Image(path=output_dir / image_name)
                    
            except ValueError:
                pass

    return AIAImages(
        aia94=images["94"],
        aia131=images["131"],
        aia171=images["171"],
        aia193=images["193"],
        aia211=images["211"],
        aia304=images["304"],
        aia335=images["335"],
        aia1600=images["1600"],
    )

SDO_HMI_LIVE_ROOT = "https://jsoc1.stanford.edu/data/hmi/images/"
SDO_HMI_LIVE_URL = "https://jsoc1.stanford.edu/data/hmi/images/image_times.json"
SDO_HMI_NAME_IC = "_Ic_4k.jpg"
SDO_HMI_NAME_IC_FLAT = "_Ic_flat_4k.jpg"
SDO_HMI_NAME_M = "_M_4k.jpg"
SDO_HMI_SNAME_M_COLOR = "_M_color_4k.jpg"

def _fetch_hmi(data_root: Path) -> HMIImages:
    output_dir = data_root / "live/raw/hmi"
    output_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(SDO_HMI_LIVE_URL)

    if r.status_code != 200:
        raise Exception("SDO server connection error: " + r.status_code)
    
    data = r.json()

    image_source = SDO_HMI_LIVE_ROOT + datetime.now().strftime("%Y/%m/%d") + "/" + data["last"] + SDO_HMI_NAME_M
    response = requests.get(image_source)
    response.raise_for_status()

    imager_path = output_dir / "hmi_m.jpg"
    with open(imager_path, "wb") as f:
        f.write(response.content)
    img = Image(path=imager_path)

    return HMIImages(
        magnetogram=img,
        bx=img, # TODO: Replace with real observation
        by=img, # TODO: Replace with real observation
        bz=build_bz(_img=img, destination=output_dir / "hmi_bz.jpg"), # TODO: Replace with more accurate representation or real observation
        dopplergram=img # TODO: Replace with real observation
    )


DSCOVR_2H_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
DSCOVR_2H_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"

def _fetch_sensors() -> SensorData:
    r = requests.get(DSCOVR_2H_MAG_URL)
    r.raise_for_status()
    data = r.json()

    # dataset contains 2-hours observations. We need only latest hour
    mag_df = pd.DataFrame(data[len(data)//2:], columns=data[0])
    mag_df["time_tag"] = pd.to_datetime(mag_df["time_tag"])
    mag_df["bx_gsm"] = pd.to_numeric(mag_df["bx_gsm"])
    mag_df["by_gsm"] = pd.to_numeric(mag_df["by_gsm"])
    mag_df["bz_gsm"] = pd.to_numeric(mag_df["bz_gsm"])

    r = requests.get(DSCOVR_2H_PLASMA_URL)
    r.raise_for_status()
    data = r.json()
    plasma_df = pd.DataFrame(data[len(data)//2:], columns=data[0])
    plasma_df["time_tag"] = pd.to_datetime(plasma_df["time_tag"])
    plasma_df["density"] = pd.to_numeric(plasma_df["density"])
    plasma_df["speed"] = pd.to_numeric(plasma_df["speed"])
    plasma_df["temperature"] = pd.to_numeric(plasma_df["temperature"])

    sensors = SensorData(
        Bx=mag_df["bx_gsm"].mean(),
        By=mag_df["by_gsm"].mean(),
        Bz=mag_df["bz_gsm"].mean(),
        V=plasma_df["speed"].mean(),
        N=plasma_df["density"].mean(),
        T=plasma_df["temperature"].mean()
    )

    return sensors


def get_observation() -> RawObservationDataPoint:
    config = get_config()

    cached_file = config.data_root / "live/observation.json"

    if not cached_file.is_file():
        observation = RawObservationDataPoint(
            timestamp=datetime.now(),
            sensors=_fetch_sensors(),
            aia= _fetch_aia(data_root=config.data_root),
            hmi=_fetch_hmi(data_root=config.data_root)
        )
        with open(cached_file, "w") as f:
            f.write(observation.model_dump_json())
    
    with open(cached_file, "r") as f:
        observation = RawObservationDataPoint.model_validate_json(f.read())

    return observation
