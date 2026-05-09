import argparse
from pathlib import Path
from datetime import datetime, timezone

from clio.dataloaders.omniloader import fetch_omni
from common.adapters import observations_to_dataframe

def _fetch_omni(output: Path, start: datetime, end: datetime):
    if output.is_file():
        print(f"File {output} aldready exists, download skipped")
        return

    out = observations_to_dataframe(fetch_omni(start=start, end=end))
    out.to_csv(output, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="OMNIWeb historical data fetch"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYYMMDD)",
        default="20230101"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYYMMDD)",
        default="20231231"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output file name",
        default=Path("./data/raw/omni.csv")
    )
    
    args = parser.parse_args()

    if not args.out.parent.is_file():
        args.out.parent.mkdir(parents=True, exist_ok=True)

    _fetch_omni(
        output=args.out,
        start=datetime.strptime(args.start, "%Y%m%d").replace(tzinfo=timezone.utc),
        end=datetime.strptime(args.end, "%Y%m%d").replace(tzinfo=timezone.utc)
    )

if __name__ == "__main__":
    main()
