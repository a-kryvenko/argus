import json
import sys
from datetime import UTC, datetime
from pathlib import Path

def ready(path: Path) -> bool:
    try:
        state = json.loads(path.read_text())
        age = (datetime.now(UTC) - datetime.fromisoformat(state["last_checked_at"])).total_seconds()
        return state.get("scheduler_started") is True and 0 <= age < 60
    except (OSError, KeyError, TypeError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(0 if len(sys.argv) == 2 and ready(Path(sys.argv[1])) else 1)
