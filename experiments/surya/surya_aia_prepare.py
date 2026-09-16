"""Download a pinned Surya checkpoint and two hourly benchmark frames."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def download(url: str, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {target.name}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            expected = response.headers.get("Content-Length")
            while block := response.read(4 * 1024 * 1024):
                out.write(block)
        if expected is not None and partial.stat().st_size != int(expected):
            raise ValueError(f"Incomplete download: {target}")
        partial.replace(target)
    with target.open("rb") as src:
        digest = hashlib.file_digest(src, "sha256").hexdigest()
    return {"path": str(target.resolve()), "url": url,
            "bytes": target.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/experiments/surya_aia"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    repo = "nasa-ibm-ai4science/Surya-1.0"
    # Retain the revision on subsequent invocations.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        revision = manifest["model_revision"]
    else:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
            revision = json.load(r)["sha"]
    assets = [download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                       root / "assets" / revision / name)
              for name in ("config.yaml", "scalers.yaml", "surya.366m.v1.pt")]
    frames = []
    for hour in (0, 1):
        name = f"20141001_{hour:02d}00.nc"
        record = download(f"https://nasa-surya-bench.s3.amazonaws.com/2014/10/{name}",
                          root / "frames" / name)
        record["timestamp"] = f"2014-10-01T{hour:02d}:00:00Z"
        frames.append(record)
    manifest = {"model_repo": repo, "model_revision": revision,
                "assets": assets, "frames": frames,
                "purpose": "Technical probe only; not a forecast evaluation dataset."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
