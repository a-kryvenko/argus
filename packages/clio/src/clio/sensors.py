import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from common.config import get_config
from common.schemas.observation import Observation, ObservationPoint

from clio.dataloaders.gong_loader import GONG_Loader
from clio.dataloaders.spdf_loader import SPDF_Loader
from clio.dataloaders.swpc_loader import SWPC_Loader
from clio.exceptions import RemoteServerException

"""
We are able to get only up to 6 day live observations directly from NOAA json products.
For older data we must re-download Archive dataset from spdf or omniweb
"""
MAX_JSON_TIME_DAYS = 6

REQUIRED_HISTORICAL_FRAME_DAYS = 30

SENSORS_MIN_INTERVAL_SECONDS = 3600
GONG_MIN_INTERVAL_SECONDS = 3600

class Sensors:
    def get_live_observations_frame() -> Observation:
        config = get_config()

        live_dataset = Sensors._get_observations_frame(config.workdir / config.project_config["paths"]["live_sensors"])

        # TODO: Implement more sophisticated missing data resolution
        live_dataset = live_dataset.fillna(live_dataset.mean(numeric_only=True))
    
        points = []
        for _, record in live_dataset.iterrows():
            points.append(ObservationPoint(
                issue_time=record["issue_time"].to_pydatetime(),
                bx=record["bx"],
                by=record["by"],
                bz=record["bz"],
                v=record["v"],
                n=record["n"],
                t=record["t"],
                kp=int(record["kp"]),
                dst=int(record["dst"]),
                ap=int(record["ap"]),
                f10_7=int(record["f10_7"])
            ))
    
        Sensors._fetch_live_gong(p=config.workdir / config.project_config["paths"]["live_gong"])
        
        return Observation(points=points)

    def _get_observations_frame(observations_path: Path) -> pd.DataFrame:
        observations_updated = False

        observations = Sensors._load_observations_file(observations_path)

        if observations is None:
            now = datetime.now(UTC)

            observations = SPDF_Loader.load(
                start_date=now - timedelta(days=REQUIRED_HISTORICAL_FRAME_DAYS),
                end_date=now - timedelta(days=MAX_JSON_TIME_DAYS - 1)
            )

            observations_updated = True

        if observations is None:
            raise RemoteServerException("Failed to download 6+ days historical frame")

        time_delta = datetime.now(UTC) - observations.iloc[-1]["issue_time"]

        if time_delta > timedelta(seconds=SENSORS_MIN_INTERVAL_SECONDS):
            """
            TODO: sensors have delay around 6 minutes. So there is problem when script executed at *:00-*:10
            Need to fix it
            """
            latest_observations = SWPC_Loader.load(start_date=observations.iloc[-1]["issue_time"])
            observations = pd.concat([observations, latest_observations])

            observations.set_index("issue_time", drop=False)
            observations = observations.sort_index()
            observations = observations.resample("1h").first()
            observations = observations.replace('', np.nan)

            observations_updated = True

        if observations_updated:
            observations.to_csv(observations_path, index=False)
        
        return observations

    def _load_observations_file(observations_path: Path) -> pd.DataFrame | None:
        if not observations_path.is_file():
            return None

        df = pd.read_csv(observations_path, parse_dates=["issue_time"])
        df = df.set_index("issue_time", drop=False)

        if df.empty:
            return None

        now = datetime.now(UTC)

        oldest = df.iloc[0]["issue_time"]
        newest = df.iloc[-1]["issue_time"]

        max_age = timedelta(days=MAX_JSON_TIME_DAYS)
        required_history = timedelta(days=REQUIRED_HISTORICAL_FRAME_DAYS - 1)

        if now - newest > max_age:
            return None

        if now - oldest < required_history:
            return None

        return df

    def _fetch_live_gong(p: Path):
        if p.exists() and (datetime.now(UTC) - datetime.fromtimestamp(p.stat().st_mtime, UTC) < timedelta(seconds=GONG_MIN_INTERVAL_SECONDS)):
            return

        try:
            tmp_path = p.parent / (p.name + ".tmp")
            GONG_Loader.load_live(tmp_path)
            shutil.move(tmp_path, p)
        except:
            pass