"""Typed AIA receipts paired with versioned extraction caches."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import io
import json

import numpy as np
import pandas as pd

from argus_prophet.services.source_cache import atomic_write


@dataclass(frozen=True)
class AIARecord:
    slot_at: datetime
    observed_at: datetime
    available_at: datetime
    sha256: str
    cache_path: Path
    b0_deg: float
    valid_fraction: float
    carrington_lon: float


def aia_record(reference, path):
    from argus_prophet.services.aia.extraction import extract_frame, VERSION
    cache = path.with_suffix(f'.features-v{VERSION}.npz')
    metadata = None
    try:
        with np.load(cache, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            if metadata['sha256'] != reference.sha256 or saved['dark'].shape != (61, 61):
                metadata = None
    except (OSError, ValueError, KeyError):
        pass
    if metadata is None:
        dark, ratio, metadata = extract_frame(path, allow_nrt=reference.source_product == 'aia.nrt_193')
        content = io.BytesIO()
        np.savez_compressed(content, dark=dark, ratio=ratio, metadata=json.dumps(metadata))
        atomic_write(cache, content.getvalue())
    observed = pd.Timestamp(metadata['observed_at']).to_pydatetime()
    if abs((observed-reference.slot_at).total_seconds()) > 600 or observed > reference.available_at:
        raise ValueError('AIA FITS timestamp conflicts with source receipt')
    return AIARecord(slot_at=reference.slot_at, observed_at=observed,
                     available_at=reference.available_at, sha256=reference.sha256,
                     cache_path=cache, b0_deg=metadata['b0_deg'],
                     valid_fraction=metadata['valid_fraction'], carrington_lon=metadata['carrington_lon'])


