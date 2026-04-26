from pathlib import Path
import pandas as pd
import datetime

from .dataloaders.omniloader import fetch_omni

def load_historical(
        start: datetime,
        end: datetime,
        output_dir: Path
):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    start_rounded = start.ceil("h")
    end_rounded = end.ceil("h")

    if end_rounded < start_rounded:
        end_rounded = start_rounded
    
    rng = pd.date_range(start=start, end=end, freq="h")

    df = pd.DataFrame({"timestamp": rng})
    omni = fetch_omni(start_rounded, end_rounded)
    
    
    