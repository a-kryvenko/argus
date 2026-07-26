from enum import Enum
from typing import Any

from forecast.inference._forecast_service import DefaultForecastService
from forecast.inference.geomagnetic_fs import APFS, DstFS, KPProbaFS
from forecast.inference.hmf_fs import HMFSouthProbaFS, HMFTotalProbaFS
from forecast.inference.plasma_fs import SWSpeedFS, SWSpeedProbaFS
from forecast.inference.radiation_fs import F107FS, M10FS, S10FS, Y10FS


class ForecastService(Enum):
    PLASMA_SPEED_QUANTILE = SWSpeedFS
    PLASMA_SPEED_THRESHOLD = SWSpeedProbaFS

    KP_INDEX_THRESHOLD = KPProbaFS
    AP_INDEX_QUANTILE = APFS
    DST_QUANTILE = DstFS

    HMF_TOTAL_THRESHOLD = HMFTotalProbaFS
    HMF_SOUTH_THRESHOLD = HMFSouthProbaFS

    F107_QUANTILE = F107FS
    S10_QUANTILE = S10FS
    M10_QUANTILE = M10FS
    Y10_QUANTILE = Y10FS

class ForecastServiceRegistry:
    @staticmethod
    def get(
        service: ForecastService,
    ) -> DefaultForecastService:
        return service.value