#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from surya_adapter.data.sdo_manifest import build_manifest_from_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple SDO file manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = build_manifest_from_directory(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
