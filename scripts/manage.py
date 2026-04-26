#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "project.yaml"

load_dotenv(".env")
if os.path.isfile(".env.local"):
    load_dotenv(".env.local")

def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid config: {path}")
    return cfg


def _resolve(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _script_cmd(script_name: str, *args: str) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts" / script_name), *args]


def _pythonpath_env(paths: list[str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if paths:
        resolved = [str(_resolve(REPO_ROOT, p)) for p in paths]
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(resolved + ([current] if current else []))
    return env


def cmd_setup(cfg: dict[str, Any]) -> None:
    for rel in [
        "data/raw/omni",
        "data/raw/dscovr",
        "data/raw/sdo",
        "data/processed",
        "data/features/surya",
        "data/artifacts/forecast",
    ]:
        p = _resolve(REPO_ROOT, rel)
        p.mkdir(parents=True, exist_ok=True)
        print(f"ok: directory {p}")

    imports = list(cfg.get("checks", {}).get("imports", {}).get("core", []))
    if cfg.get("surya", {}).get("enabled", False):
        imports += list(cfg.get("checks", {}).get("imports", {}).get("surya", []))

    missing: list[str] = []
    for name in imports:
        try:
            __import__(name)
        except Exception:
            missing.append(name)

    if missing:
        print("missing imports:", ", ".join(missing))
        print("setup incomplete")
        sys.exit(2)

    critical_files = [
        cfg.get("surya", {}).get("vendor_root"),
        cfg.get("surya", {}).get("config_path"),
        cfg.get("surya", {}).get("checkpoint_path"),
        cfg.get("surya", {}).get("pretrained_path"),
        cfg.get("surya", {}).get("scalers_path"),
    ]
    for rel in critical_files:
        if rel:
            p = _resolve(REPO_ROOT, rel)
            print(f"{'ok' if p.exists() else 'missing'}: {p}")

    print("setup ok")
    print("next: uv run python scripts/manage.py prepare-forecast")


def cmd_prepare_forecast(cfg: dict[str, Any]) -> None:
    sources = cfg.get("sources", {})
    outputs = cfg.get("outputs", {})
    forecast = cfg.get("forecast", {})
    surya = cfg.get("surya", {})

    omni = sources.get("omni", {})
    if omni.get("enabled", False):
        _run(_script_cmd(
            "sync_omni.py",
            "--start", str(omni["start"]),
            "--end", str(omni["end"]),
            "--out", str(_resolve(REPO_ROOT, omni["out"])),
        ))

    dscovr = sources.get("dscovr", {})
    if dscovr.get("enabled", False):
        _run(_script_cmd(
            "sync_dscovr.py",
            "--range", str(dscovr["range"]),
            "--out", str(_resolve(REPO_ROOT, dscovr["out"])),
        ))

    merged_csv = _resolve(REPO_ROOT, outputs["merged_csv"])
    _run(_script_cmd(
        "build_forecast_dataset.py",
        # "--omni", str(_resolve(REPO_ROOT, omni["out"])),
        "--dscovr", str(_resolve(REPO_ROOT, dscovr["out"])),
        "--out-csv", str(merged_csv),
    ))

    sdo = sources.get("sdo", {})
    if sdo.get("enabled", False):
        _run(_script_cmd(
            "sync_sdo.py",
            "--email", str(os.getenv("SDO_EMAIL")),
            "--timestamps-csv", str(_resolve(REPO_ROOT, sdo["timestamps_csv"])),
            "--out-dir", str(_resolve(REPO_ROOT, sdo["out_dir"])),
            "--manifest-out", str(_resolve(REPO_ROOT, sdo["manifest_out"])),
        ))

    if surya.get("enabled", False):
        # _run(_script_cmd(
        #     "export_surya_embeddings.py",
        #     "--config", str(_resolve(REPO_ROOT, "configs/surya/export_embeddings.yaml")),
        #     "--device", str(surya.get("device", "cpu")),
        #     "--num-samples", str(surya.get("num_samples", 8)),
        # ))
        pass

    _run(_script_cmd(
        "build_forecast_dataset.py",
        "--omni", str(_resolve(REPO_ROOT, omni["out"])),
        "--dscovr", str(_resolve(REPO_ROOT, dscovr["out"])),
        "--out-csv", str(merged_csv),
        "--build-npz",
        "--npz-out", str(_resolve(REPO_ROOT, outputs["forecast_dataset_npz"])),
        "--embeddings-path", str(_resolve(REPO_ROOT, surya["output_path"])),
        "--history-hours", str(forecast["history_hours"]),
        "--forecast-hours", str(forecast["forecast_hours"]),
        "--embedding-lag-hours", str(forecast["embedding_lag_hours"]),
        "--max-embedding-age-hours", str(forecast["max_embedding_age_hours"]),
        *( ["--require-embeddings"] if forecast.get("require_embeddings", False) else [] ),
    ))

    _run([
        sys.executable,
        "-m",
        "forecast_core.data_pipelines.omni_dscovr_sdo.compute_scalers",
        "--input", str(_resolve(REPO_ROOT, outputs["forecast_dataset_npz"])),
        "--output", str(_resolve(REPO_ROOT, outputs["forecast_scalers_yaml"])),
    ])

    print("prepare-forecast ok")
    print("next: uv run python scripts/manage.py serve-public")


def cmd_serve_public(cfg: dict[str, Any]) -> None:
    runtime = cfg.get("runtime", {})
    env = _pythonpath_env(runtime.get("api_pythonpath"))
    _run([
        sys.executable,
        "-m",
        "uvicorn",
        str(runtime.get("api_module", "api_public.main:app")),
        "--host", str(runtime.get("api_host", "127.0.0.1")),
        "--port", str(runtime.get("api_port", 8000)),
        "--reload",
    ], env=env)


def cmd_forecast_smoke(cfg: dict[str, Any]) -> None:
    import pandas as pd
    import urllib.request

    outputs = cfg.get("outputs", {})
    runtime = cfg.get("runtime", {})
    forecast = cfg.get("forecast", {})

    merged_csv = _resolve(REPO_ROOT, outputs["merged_csv"])
    payload_path = _resolve(REPO_ROOT, outputs["smoke_payload_json"])

    df = pd.read_csv(merged_csv)
    history = df.tail(int(forecast.get("history_hours", 48))).copy()
    history["timestamp"] = history["timestamp"].astype(str)

    payload = {
        "reference_timestamp": str(history.iloc[-1]["timestamp"]),
        "history": history.to_dict(orient="records"),
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload))

    url = f"http://{runtime.get('api_host', '127.0.0.1')}:{runtime.get('api_port', 8000)}/forecast"
    req = urllib.request.Request(
        url,
        data=payload_path.read_bytes(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode()
        print(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus operator entrypoint")
    parser.add_argument("command", choices=["setup", "prepare-forecast", "serve-public", "forecast-smoke"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())

    if args.command == "setup":
        cmd_setup(cfg)
    elif args.command == "prepare-forecast":
        cmd_prepare_forecast(cfg)
    elif args.command == "serve-public":
        cmd_serve_public(cfg)
    elif args.command == "forecast-smoke":
        cmd_forecast_smoke(cfg)


if __name__ == "__main__":
    main()
