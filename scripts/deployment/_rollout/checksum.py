"""Compute the checksum shared by bundle creation and deployment."""
import hashlib
from pathlib import Path


def get_checksum(folder: Path) -> str:
    md5 = hashlib.md5()

    for file in sorted(folder.rglob("*")):
        if file.is_file() and '__pycache__' not in file.parts:
            relative_path = file.relative_to(folder)
            md5.update(str(relative_path).encode())

            with file.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    md5.update(chunk)

    return md5.hexdigest()
