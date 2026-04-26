from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def run_vendor_image_forecast(vendor_root: str | Path, *, config_path: str | Path, checkpoint_path: str | Path, output_dir: str | Path, device: str = "cpu") -> int:
    vendor_root = Path(vendor_root)
    script = vendor_root / "downstream_examples" / "solar_wind_forecasting" / "infer.py"
    cmd = [sys.executable, str(script), "--config_path", str(config_path), "--checkpoint_path", str(checkpoint_path), "--output_dir", str(output_dir), "--device", device]
    return subprocess.call(cmd, cwd=vendor_root)
