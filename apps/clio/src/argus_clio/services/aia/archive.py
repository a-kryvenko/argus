"""Archive location shared by AIA ingestion and feature reads."""
import os
from pathlib import Path

from common.config import get_config


def archive_root() -> Path:
    default = get_config().data_root / 'observations/aia193'
    return Path(os.getenv('ARGUS_AIA_ARCHIVE', str(default)))
