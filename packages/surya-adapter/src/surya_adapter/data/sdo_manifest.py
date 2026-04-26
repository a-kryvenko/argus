from __future__ import annotations

from pathlib import Path
import pandas as pd

EXPECTED_CHANNELS = [
    'aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304', 'aia335', 'aia1600',
    'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v'
]


def build_manifest_from_directory(root: str | Path, pattern: str = "**/*") -> pd.DataFrame:
    root = Path(root)
    rows = []
    for p in root.glob(pattern):
        if p.is_file():
            rows.append({"path": str(p)})
    return pd.DataFrame(rows)
