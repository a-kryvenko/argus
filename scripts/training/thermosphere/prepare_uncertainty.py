"""Download disjoint density calibration data or fit frozen upper ap scenarios.

Run --download (network) once, then --train (offline). Nothing is fitted on 2025.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time

from prepare import HERE, ROOT, digest, download
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"data/.matplotlib"))
import joblib
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor

from train_ap import CONFIG as AP_CONFIG, OUT as MEDIAN_DIR, FEATURES, causal_features, read_omni, training_rows

CONFIG = json.loads((HERE/"config_uncertainty.json").read_text())
OUT = ROOT/"data/experiments"/CONFIG["experiment"]


def download_calibration():
    dates = [d for start, end in CONFIG["calibration_windows"] for d in pd.date_range(start, end)]
    def fetch(day):
        stamp = day.strftime("%Y%m%d")
        name = f"GF_OPER_DNS1ACC_2__{stamp}T000000_{stamp}T235959_0003.cdf"
        remote = f"swarm/Multimission/GRACE-FO/DNS/Sat_1/{day.year}/{name}"
        url = requests.Request("GET", "https://swarm-diss.eo.esa.int/",
            params={"do": "download", "file": remote}).prepare().url
        item = download(url, OUT/"gracefo"/name)
        print(f"Calibration data: {day.date()}", flush=True)
        return item
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(fetch, dates))
    (OUT/"download_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def train_upper():
    median_info = json.loads((MEDIAN_DIR/"ap_training.json").read_text())
    if digest(MEDIAN_DIR/"ap_q50.joblib") != median_info["artifact_sha256"]:
        raise ValueError("Median model changed")
    raw, paths = read_omni()
    rows = training_rows(raw, causal_features(raw))
    matrix_hash = hashlib.sha256(pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes()).hexdigest()
    if matrix_hash != median_info["training_matrix_sha256"]:
        raise ValueError("Upper scenarios must use the same pre-test training rows as q50")
    models, metadata = {}, {}
    for alpha in CONFIG["upper_ap_quantiles"]:
        start = time.monotonic()
        name = f"q{int(alpha*100)}"
        parameters = {**AP_CONFIG["model_params"], "alpha": alpha}
        model = LGBMRegressor(**parameters)
        print(f"Fit ap {name}: {len(rows):,} rows", flush=True)
        model.fit(rows[FEATURES].astype(np.float32), rows.target_ap.to_numpy(dtype=np.float32))
        info = {**median_info, "quantile": alpha, "parameters": parameters,
                "upper_training_code_sha256": digest(HERE/"prepare_uncertainty.py")}
        # The copied artifact hash describes q50; the upper bundle gets its own hash.
        info.pop("artifact_sha256")
        bundle = {"model": model, "metadata": info}
        joblib.dump(bundle, OUT/f"ap_{name}.joblib")
        metadata[name] = {"artifact_sha256": digest(OUT/f"ap_{name}.joblib"),
            "parameters": parameters, "max_label_available_at": rows.label_available_at.max().isoformat(),
            "elapsed_seconds": time.monotonic()-start}
        print(f"{name} fitted in {time.monotonic()-start:.1f}s", flush=True)
    record = {"models": metadata, "config": CONFIG, "training_matrix_sha256": matrix_hash,
        "median_artifact_sha256": median_info["artifact_sha256"],
        "source_files": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "code_sha256": digest(HERE/"prepare_uncertainty.py")}
    (OUT/"training.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_calibration()
    if args.train:
        train_upper()
