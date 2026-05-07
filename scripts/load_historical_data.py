import argparse
from pathlib import Path
from datetime import datetime, timezone
from clio.dataloaders.omniloader import fetch_omni

import pandas as pd
import numpy as np

def _fetch_omni(output: Path, start: datetime, end: datetime):
    if output.is_file():
        return
    
    df = fetch_omni(start=start, end=end)

    out = pd.DataFrame()
    out["time"] = df["timestamp"]
    out["v_obs"] = df["V"]
    out["n_obs"] = df["N"]
    out["bz_obs"] = df["BZ_GSM"]

    # Approximate total IMF magnitude from available GSM/GSE components.
    # Good enough for v1 context feature.
    out["bt_obs"] = np.sqrt(
        df["BX_GSM"] ** 2 +
        df["BY_GSM"] ** 2 +
        df["BZ_GSM"] ** 2
    )

    out["kp"] = df["KP_10"]

    out["temperature"] = df["T"]

    out = out[["time", "v_obs", "n_obs", "bz_obs", "bt_obs", "kp", "temperature"]]
    out = out.dropna(subset=["time", "v_obs"])
    out = out.sort_values("time")

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
        default=Path("./data/historical/omni.csv")
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
