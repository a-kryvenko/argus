"""Display images from NASA's AIA browse products, separate from scientific FITS."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

CHANNELS = ('094', '131', '171', '193', '211', '304', '335', '1600', '1700')
SIZES = (128, 256, 512, 1024, 2048, 4096)
ASSET_PATTERN = r'\d{10}_(' + '|'.join(CHANNELS) + r')_[0-9a-f]{64}'
IMAGE_PATTERN = ASSET_PATTERN + r'_(128|256|512|1024|2048|4096)\.(jpg|webp)'
CACHE_CONTROL = 'public, max-age=31536000, immutable'


class SDOBrowseImage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    channel: Literal['094', '131', '171', '193', '211', '304', '335', '1600', '1700']
    asset_id: str = Field(pattern='^' + ASSET_PATTERN + '$')
    captured_at: AwareDatetime
    source_updated_at: AwareDatetime | None = None


class SDOBrowseCatalog(BaseModel):
    model_config = ConfigDict(extra='forbid')
    images: list[SDOBrowseImage]
