"""Download pinned research inputs and build unmodified DTM2020 locally.

Run from repository root: .venv/bin/python scripts/training/thermosphere/prepare.py
Requires requests, a Fortran compiler and permission to use the upstream license.
No model sources or coefficients are redistributed in this repository.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((HERE / "config.json").read_text())
OUT = ROOT / "data/experiments" / CONFIG["experiment"]
FILES = {
    "src/dtm2020/dtm2020_F107_Kp-subr.f90": "18d7b736071802d6f7cdf65d1ae6c8b5e7f0dfca8f163390abc035f14db5ae72",
    "src/dtm2020/Mod_DTM2020F107Kp_Benchmark.f90": "6d121041453d112d221de1668c5d4b7d2dda9dec66ea1d0b6dcc5a5b81448907",
    "data/DTM_2020_F107_Kp.dat": "c08f7ca61a3521a736a857850bd6cacd169f9302c356dc8523608b1a1d941588",
    "LICENSE": "b45cc69494a849efd7232d3e2de85480c8ac6e389c974244a9b882aeab4e5085",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(url, timeout=(15, 90))
        response.raise_for_status()
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    actual = digest(path)
    if expected and actual != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return {"url": url, "path": str(path.relative_to(ROOT)), "sha256": actual,
            "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=shutil.which("gfortran"))
    args = parser.parse_args()
    base = f"https://raw.githubusercontent.com/swami-h2020-eu/mcm/{CONFIG['upstream_commit']}/"
    manifest = [download(base + name, OUT / "upstream" / Path(name).name, sha)
                for name, sha in FILES.items()]
    dates = [day for start, end in CONFIG["windows"] for day in pd.date_range(start, end)]

    def observation(day):
        stamp = day.strftime("%Y%m%d")
        name = f"GF_OPER_DNS1ACC_2__{stamp}T000000_{stamp}T235959_{CONFIG['density_product_version']}.cdf"
        remote = f"swarm/Multimission/GRACE-FO/DNS/Sat_1/{day.year}/{name}"
        url = requests.Request("GET", "https://swarm-diss.eo.esa.int/",
                               params={"do": "download", "file": remote}).prepare().url
        result = download(url, OUT / "gracefo" / name)
        print(f"Ready {day.date()}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest.extend(pool.map(observation, dates))
    (OUT / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    compiler = args.compiler
    extra = []
    if not compiler:
        local = OUT / "compiler/root/usr/bin/x86_64-linux-gnu-gfortran-13"
        if not local.exists():
            raise RuntimeError("Install gfortran or supply --compiler; downloads are complete")
        compiler = str(local)
        extra = ["-B", "/usr/lib/gcc/x86_64-linux-gnu/13/", "-B",
                 str(OUT / "compiler/root/usr/lib/gcc/x86_64-linux-gnu/13") + "/"]
    command = [compiler, *extra, "-O2", "-fPIC", "-shared", "-fdefault-real-8",
               str(OUT / "upstream/dtm2020_F107_Kp-subr.f90"),
               str(HERE / "dtm_bridge.f90"), "-o", str(OUT / "libdtm2020.so")]
    subprocess.run(command, check=True)
    (OUT / "build.json").write_text(json.dumps({
        "command": command, "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "bridge_sha256": digest(HERE / "dtm_bridge.f90"),
        "library_sha256": digest(OUT / "libdtm2020.so"),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
