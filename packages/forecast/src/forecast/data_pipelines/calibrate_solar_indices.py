"""Fit a frozen calibration from local, overlapping GOES and SOLFSMY data."""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from forecast.data_pipelines.solar_indices import (
    fit_solar_index_calibrations,
    save_solar_index_calibrations,
)


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--goes", required=True, type=Path, help="Normalized GOES CSV/parquet")
    parser.add_argument("--solfsmy", required=True, type=Path, help="Parsed SOLFSMY CSV/parquet")
    parser.add_argument("--output", required=True, type=Path, help="Output calibration JSON")
    args = parser.parse_args()
    calibrations = fit_solar_index_calibrations(
        _read_frame(args.goes), _read_frame(args.solfsmy),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_solar_index_calibrations(calibrations, args.output)
    print(f"Saved frozen solar-index calibration to {args.output}")


if __name__ == "__main__":
    main()
