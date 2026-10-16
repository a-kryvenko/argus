"""Atomic publication of versioned JSON bundles."""
import json
import os
from pathlib import Path
import tempfile


def atomic_publish(bundle, output):
    content = json.dumps(bundle, allow_nan=False, separators=(',', ':')).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A fully written immutable version is retained before the live pointer is replaced.
    version = output.parent / f"{bundle['version']}.json"
    if version.exists():
        raise ValueError('Demo version already exists')
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, version)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

