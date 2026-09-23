"""Local cache locations and file provenance."""
import os,hashlib
from pathlib import Path
os.environ.setdefault("SUNPY_CONFIGDIR", "/tmp/argus-aia-sunpy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/argus-aia-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/argus-aia-cache")
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
