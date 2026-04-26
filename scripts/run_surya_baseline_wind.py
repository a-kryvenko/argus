#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vendor Surya downstream solar wind inference.")
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/surya"))
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    script = args.vendor_root / "downstream_examples" / "solar_wind_forecasting" / "infer.py"
    cmd = [sys.executable, str(script), "--config_path", str(args.config_path), "--checkpoint_path", str(args.checkpoint_path), "--output_dir", str(args.output_dir), "--device", args.device]
    raise SystemExit(subprocess.call(cmd, cwd=args.vendor_root))


if __name__ == "__main__":
    main()
