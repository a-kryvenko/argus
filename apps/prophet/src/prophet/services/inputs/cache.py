"""Read immutable source objects in memory; write atomic model working files."""
import hashlib
import os
from pathlib import Path
import tempfile

from common.data.datamanager import read_reference



def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def original(reference):
    content = read_reference(reference.file)
    if hashlib.sha256(content).hexdigest() != reference.sha256:
        raise ValueError('Observation checksum differs from its receipt')
    return content
