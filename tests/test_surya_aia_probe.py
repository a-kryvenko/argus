"""Small fixtures check missing-channel semantics and causal frame alignment."""

import importlib.util
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "surya_aia_probe", Path(__file__).resolve().parents[1] / "experiments/surya/surya_aia_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def sample(tmp_path):
    # Permuted channels catch positional masking; input files contain NO HMI.
    channels = list(reversed(probe.AIA + probe.HMI))
    config = {"model": {"img_size": 4}, "data": {
        "sdo_channels": channels, "time_delta_input_minutes": [-60, 0]}}
    scalers = {c: {"sl_scale_factor": .01, "mean": .2, "std": .5, "epsilon": 1e-8}
               for c in channels}
    frames = []
    for hour in (0, 1):
        path = tmp_path / f"{hour}.nc"
        with netCDF4.Dataset(path, "w") as nc:
            nc.createDimension("y", 4)
            nc.createDimension("x", 4)
            for channel in probe.AIA:
                nc.createVariable(channel, "f4", ("y", "x"))[:] = 10 + hour
        frames.append({"path": str(path), "timestamp": f"2025-01-01T0{hour}:00:00Z"})
    return frames, config, scalers


def test_aia_only_never_requires_hmi_and_masks_in_normalized_space(sample):
    frames, config, scalers = sample
    data, dt, _ = probe.load_inputs(frames, config, scalers, "aia")
    for i, channel in enumerate(config["data"]["sdo_channels"]):
        if channel in probe.HMI:
            assert torch.count_nonzero(data[:, i]) == 0
        else:
            expected = (np.log1p(.1) - .2) / (.5 + 1e-8)
            assert data[0, i, 0, 0, 0].item() == pytest.approx(expected)
    assert dt.tolist() == [[1., 0.]]


def test_incorrect_time_gap_is_rejected(sample):
    frames, config, scalers = sample
    frames[0]["timestamp"] = "2024-12-31T23:00:00Z"
    with pytest.raises(ValueError, match="cadence"):
        probe.load_inputs(frames, config, scalers, "aia")


def test_missing_aia_is_not_silently_zero_filled(sample):
    frames, config, scalers = sample
    with netCDF4.Dataset(frames[0]["path"], "a") as nc:
        nc.renameVariable("aia94", "missing94")
    with pytest.raises(KeyError):
        probe.load_inputs(frames, config, scalers, "aia")


def test_nonfinite_pixels_rejected(sample):
    frames, config, scalers = sample
    with netCDF4.Dataset(frames[0]["path"], "a") as nc:
        nc.variables["aia94"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        probe.load_inputs(frames, config, scalers, "aia")


def test_pooling_retains_spatial_quadrants():
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    result = probe.pool_tokens(tokens, 4)
    assert result["global_mean"].tolist() == [[7.5]]
    assert result["grid_2x2"].tolist() == [[2.5, 4.5, 10.5, 12.5]]


def test_checkpoint_load_is_strict_and_returns_encoder_tokens(tmp_path):
    # Tiny real Surya architecture; a saved random state only checks wiring.
    if not (probe.ROOT / "vendor/surya/surya").is_dir():
        pytest.skip("Optional vendor/surya checkout is not installed")
    import sys
    sys.path.insert(0, str(probe.ROOT / "vendor/surya"))
    from surya.models.helio_spectformer import HelioSpectFormer
    model_opts = dict(img_size=32, patch_size=8, embed_dim=16, depth=2,
                      n_spectral_blocks=1, num_heads=2, mlp_ratio=2.,
                      drop_rate=0., window_size=2, dp_rank=2,
                      learned_flow=False, rpe=False, ensemble=None, finetune=False)
    config = {"model": model_opts, "data": {"sdo_channels": list(probe.AIA + probe.HMI),
              "time_delta_input_minutes": [-60, 0]}}
    model = HelioSpectFormer(**model_opts, in_chans=13,
                            time_embedding={"type": "linear", "time_dim": 2})
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    encoder = probe.build_encoder(config, path)
    assert not hasattr(encoder, "unembed")
    assert not any(p.requires_grad for p in encoder.parameters())
    with torch.inference_mode():
        result = encoder({"ts": torch.zeros(1, 13, 2, 32, 32), "time_delta_input": torch.tensor([[1., 0.]])})
    assert result.shape == (1, 16, 16)
    state = model.state_dict()
    del state["embedding.patch_embed.proj.weight"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        probe.build_encoder(config, path)


def test_manifest_timestamp_must_match_file_metadata(sample):
    frames, config, scalers = sample
    with netCDF4.Dataset(frames[0]["path"], "a") as nc:
        nc.data_time = "20250102_0000"
    with pytest.raises(ValueError, match="data_time"):
        probe.load_inputs(frames, config, scalers, "aia")
