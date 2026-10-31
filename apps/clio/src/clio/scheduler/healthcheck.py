import json
import sys
from datetime import UTC, datetime
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
age = (datetime.now(UTC) - datetime.fromisoformat(state["last_checked_at"])).total_seconds()
sys.exit(0 if 0 <= age < 60 else 1)