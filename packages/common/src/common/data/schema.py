from pydantic import BaseModel, Field
from typing import List, Tuple


class AIAConfig(BaseModel):
    wavelengths: List[str] = Field(..., description="AIA channels")

class HMIConfig(BaseModel):
    components: List[str] = Field(..., description="HMI components")


class SolarWindConfig(BaseModel):
    features: List[str] = Field(..., description="L1 sensor features")


class DatasetConfig(BaseModel):
    aia: AIAConfig
    hmi: HMIConfig
    solar_wind: SolarWindConfig

    image_shape: Tuple[int, int]
    dtype: str = "float32"
    schema_version: str = "1.0"