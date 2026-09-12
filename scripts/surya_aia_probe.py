"""Frozen Surya encoder probe on curated, native-resolution SuryaBench NetCDFs.

This measures execution, not forecasting skill. No raw-FITS preprocessing is implied.
"""

import argparse
from contextlib import nullcontext
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import hdf5plugin  # noqa: F401 -- registers NetCDF/HDF5 compression filters
import netCDF4
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
AIA = ("aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600")
HMI = ("hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v")


def normalize(raw: np.ndarray, scaler: dict) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    scaled = raw * np.float32(scaler["sl_scale_factor"])
    return ((np.sign(scaled) * np.log1p(np.abs(scaled)) - np.float32(scaler["mean"]))
            / (np.float32(scaler["std"]) + np.float32(scaler["epsilon"])))


def load_inputs(frames: list[dict], config: dict, scalers: dict, mode: str):
    channels = config["data"]["sdo_channels"]
    if set(channels) != set(AIA + HMI) or len(channels) != 13:
        raise ValueError("Expected exactly the 13 named Surya AIA/HMI channels")
    offsets = config["data"]["time_delta_input_minutes"]
    if len(frames) != len(offsets) or offsets != sorted(offsets) or offsets[-1] != 0:
        raise ValueError("Frames must match ordered checkpoint input offsets ending at zero")
    times = [datetime.fromisoformat(f["timestamp"].replace("Z", "+00:00")) for f in frames]
    if any(t.tzinfo is None for t in times):
        raise ValueError("Frame timestamps must specify timezone")
    actual = [(t - times[-1]).total_seconds() / 60 for t in times]
    if actual != offsets:
        raise ValueError(f"Input cadence mismatch: {actual} != {offsets}")
    size = config["model"]["img_size"]
    # Zero is a normalized-space ablation, not an observed zero magnetic field.
    data = np.zeros((1, len(channels), len(frames), size, size), dtype=np.float32)
    kept = set(AIA) | ({"hmi_m"} if mode == "aia-los" else set())
    if mode == "full":
        kept = set(channels)
    quality = []
    for ti, frame in enumerate(frames):
        with netCDF4.Dataset(frame["path"]) as nc:
            if "data_time" in nc.ncattrs():
                expected_time = times[ti].astimezone(UTC).strftime("%Y%m%d_%H%M")
                if nc.getncattr("data_time") != expected_time:
                    raise ValueError(f"NetCDF data_time does not match manifest: {frame['path']}")
            for ci, channel in enumerate(channels):
                if channel not in kept:
                    continue  # Never read absent HMI channels, even if they exist in the file.
                values = np.ma.asarray(nc.variables[channel][:], dtype=np.float32)
                if values.shape != (size, size):
                    raise ValueError(f"{channel}: {values.shape}; expected {(size, size)}. No resizing allowed.")
                raw = values.filled(np.nan)
                bad = ~np.isfinite(raw)
                if bad.any():
                    raise ValueError(f"{channel}: {bad.sum()} missing/nonfinite pixels; inspect preprocessing")
                data[0, ci, ti] = normalize(raw, scalers[channel])
                quality.append({"timestamp": frame["timestamp"], "channel": channel,
                                "min": float(raw.min()), "max": float(raw.max()),
                                "source_attrs": {key: str(nc.variables[channel].getncattr(key))
                                                 for key in ("unit", "t_obs", "qflag")
                                                 if key in nc.variables[channel].ncattrs()}})
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite normalized inputs")
    return torch.from_numpy(data), torch.tensor([[-v / 60 for v in offsets]], dtype=torch.float32), quality


def pool_tokens(tokens: torch.Tensor, grid: int) -> dict[str, torch.Tensor]:
    if tokens.ndim != 3 or tokens.shape[1] != grid * grid or grid % 2:
        raise ValueError("Expected a square, even spatial token grid without extra tokens")
    spatial = tokens.reshape(tokens.shape[0], grid, grid, tokens.shape[-1])
    quadrants = [spatial[:, y:y + grid // 2, x:x + grid // 2].mean((1, 2))
                 for y in (0, grid // 2) for x in (0, grid // 2)]
    return {"global_mean": tokens.mean(1), "grid_2x2": torch.cat(quadrants, dim=-1)}


def build_encoder(config: dict, checkpoint: Path):
    sys.path.insert(0, str(ROOT / "vendor" / "surya"))
    from surya.models.helio_spectformer import HelioSpectFormer

    opts = dict(config["model"])
    if opts.get("learned_flow") or opts.get("ensemble") or opts.get("finetune"):
        raise ValueError("Probe requires the deterministic foundation checkpoint without learned flow")
    opts.update(in_chans=len(config["data"]["sdo_channels"]),
                time_embedding={"type": "linear", "time_dim": len(config["data"]["time_delta_input_minutes"])},
                init_weights=False, checkpoint_layers=None)
    model = HelioSpectFormer(**opts)
    # Load the complete model strictly before removing the image decoder.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    model.finetune = True
    del model.unembed
    model.requires_grad_(False)
    return model.eval()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/experiments/surya_aia/manifest.json"))
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--mode", choices=("aia", "aia-los", "full"), default="aia")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 1 or args.cpu_threads < 1:
        parser.error("repeats and cpu-threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use --device cpu explicitly or run on a GPU host.")
    device = torch.device(args.device)
    if args.precision == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA device does not support bfloat16")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    manifest = json.loads(args.manifest.read_text())
    assets = {Path(a["path"]).name: Path(a["path"]) for a in manifest["assets"]}
    # Verify provenance, including cached files, before expensive execution.
    for record in manifest["assets"] + manifest["frames"]:
        with Path(record["path"]).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {record['path']}")
    config = yaml.safe_load(assets["config.yaml"].read_text())
    scalers = yaml.safe_load(assets["scalers.yaml"].read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "purpose": "technical_probe_not_forecast_skill",
              "mode": args.mode, "device": str(device), "precision": args.precision,
              "torch": torch.__version__, "cpu_threads": args.cpu_threads,
              "model_revision": manifest["model_revision"], "manifest": manifest,
              "masked_channels": [c for c in HMI if args.mode == "aia" or (args.mode == "aia-los" and c != "hmi_m")],
              "mask_space": "normalized_zero", "config": config,
              "surya_git_revision": subprocess.check_output(
                  ["git", "-C", str(ROOT / "vendor/surya"), "rev-parse", "HEAD"], text=True).strip(),
              "started_utc": datetime.now(UTC).isoformat()}
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        start = perf_counter()
        print("Loading encoder", flush=True)
        model = build_encoder(config, assets["surya.366m.v1.pt"]).to(device)
        report["encoder_parameters"] = sum(p.numel() for p in model.parameters())
        report["model_load_seconds"] = perf_counter() - start
        print("Loading and normalizing input frames", flush=True)
        start = perf_counter()
        inputs, dt, report["input_quality"] = load_inputs(manifest["frames"], config, scalers, args.mode)
        report["input_shape"] = list(inputs.shape)
        inputs, dt = inputs.to(device), dt.to(device)
        synchronize(device)
        report["input_load_seconds"] = perf_counter() - start
        report["forward_seconds"] = []
        # Block timing also provides progress on CPU; hooks don't change outputs.
        block_timings = []
        previous = [perf_counter()]
        def mark_block(_module, _args, _output):
            synchronize(device)
            now = perf_counter()
            block_timings.append(now - previous[0])
            previous[0] = now
            print(f"  encoder block {len(block_timings)} complete", flush=True)
        hooks = [b.register_forward_hook(mark_block) for b in
                 list(model.backbone.blocks_spectral_gating) + list(model.backbone.blocks_attention)]
        for iteration in range(args.repeats):
            print(f"Forward {iteration + 1}/{args.repeats}", flush=True)
            synchronize(device)
            start = previous[0] = perf_counter()
            context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.precision == "bfloat16" else nullcontext()
            with torch.inference_mode(), context:
                tokens = model({"ts": inputs, "time_delta_input": dt})
                if not torch.isfinite(tokens).all():
                    raise ValueError("Nonfinite encoder output")
                pooled = pool_tokens(tokens, config["model"]["img_size"] // config["model"]["patch_size"])
            synchronize(device)
            report["forward_seconds"].append(perf_counter() - start)
            report["token_shape"] = list(tokens.shape)
            arrays = {key: value.float().cpu().numpy() for key, value in pooled.items()}
            del tokens, pooled
        for hook in hooks:
            hook.remove()
        np.savez_compressed(args.output / "embeddings.npz", **arrays)
        report["embedding_shapes"] = {key: list(value.shape) for key, value in arrays.items()}
        report["block_seconds_including_embedding_before_first"] = block_timings
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if device.type == "cuda":
            report["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            report["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "forward_seconds", "peak_process_rss_mib", "embedding_shapes")}), flush=True)


if __name__ == "__main__":
    main()
