import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

def read_state(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def write_state(path: Path, state):
    state["last_checked_at"] = datetime.now(UTC).isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def needs_warmup(previous):
    if not isinstance(previous, dict) or previous.get("warmup_pending") is True:
        return True

    try:
        elapsed = datetime.now(UTC) - datetime.fromisoformat(previous["last_checked_at"])
    except (KeyError, TypeError, ValueError):
        return True
    return not timedelta() <= elapsed <= timedelta(minutes=1)


async def heartbeat(path: Path, state):
    while True:
        await asyncio.sleep(10)
        write_state(path, state)
