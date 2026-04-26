import datetime as dt
from pathlib import Path
from clio.loader import load_historical

start = dt.datetime.fromisoformat("2026-02-13T00:00:00Z")
end = dt.datetime.fromisoformat("2026-02-13T01:00:00Z")
output = Path("data/raw/")

load_historical(start, end, output)