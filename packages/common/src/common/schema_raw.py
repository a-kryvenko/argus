from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from typing import List
import datetime

class SensorData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    Bx: float
    By: float
    Bz: float
    V: float
    N: float
    T: float

class Image(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: HttpUrl

class AIAImages(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aia94: Image
    aia131: Image
    aia171: Image
    aia193: Image
    aia211: Image
    aia304: Image
    aia335: Image
    aia1600: Image

class HMIImages(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bx: Image
    by: Image
    bz: Image
    dopplergram: Image
    magnetogram: Image

class RawObservationDataPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    sensors: SensorData
    aia: AIAImages
    hmi: HMIImages

class ObservationTimeSeries(BaseModel):
    data: List[RawObservationDataPoint]