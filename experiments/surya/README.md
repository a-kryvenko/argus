# Surya AIA-only technical probe

The first stage measures whether the frozen foundation encoder can process AIA-only
inputs at native resolution, and its execution time and memory. It does **not**
measure wind forecast skill or establish that missing HMI is harmless.

## Findings before execution

- The published config has 13 channels, two input frames at offsets `[-60, 0]`
  minutes, 4096×4096 images, 16×16 patches, and 1280-dimensional tokens.
- Its `drop_hmi_probability` is **0.0**. The locally vendored implementation of
  `RandomChannelMaskerTransform` only zeros `masked_tensor[-1, ...]`, i.e. `hmi_v`
  for this channel order. It does not mask all five HMI channels.
- The probe therefore explicitly omits HMI reads and supplies zero in normalized
  space for **all five named HMI channels**. This is an ablation; robustness to it
  has not been demonstrated. Zero raw magnetic field is not the same operation.
- `HelioSpectFormer(..., finetune=True)` returns encoder tokens. We first load the
  complete checkpoint strictly, then enable this path and remove the image
  decoder. No backbone weights are adapted or resized.
- The host used for the recorded probe exposed no working CUDA device. CPU timing is useful for
  feasibility only and cannot be used to estimate GPU throughput or VRAM.

## Prepare and run

Run from the repository root, with the existing `.venv` and `vendor/surya` checkout.
The preparation script pins the Hugging Face revision, downloads the checkpoint
and two curated SuryaBench frames (2014-10-01 00:00 and 01:00), and records SHA256
checksums in a manifest. These historical frames may overlap pretraining and are
**only suitable for this execution probe**, not for a held-out skill estimate.

```bash
.venv/bin/python experiments/surya/surya_aia_prepare.py
.venv/bin/python experiments/surya/surya_aia_probe.py \
  --device cpu --output data/experiments/surya_aia/probe_cpu
```

For a GPU measurement, use a new output directory:

```bash
.venv/bin/python experiments/surya/surya_aia_probe.py \
  --device cuda --precision bfloat16 --repeats 3 \
  --output data/experiments/surya_aia/probe_cuda
```

The first forward is cold; subsequent timings are warm. Reported forward times
include encoder execution, finite-output validation and pooling, but exclude
reading/normalization and writing results. Block timings include embedding time
before the first block. CUDA hooks synchronize to make these measurements explicit.
Peak RSS is for the whole process, including model loading; CUDA peaks cover model
transfer, input tensors and execution. Float32 and bfloat16 are separate experiments.

Outputs:

- `embeddings.npz`: `global_mean` (1×1280) and `grid_2x2` (1×5120).
  Quadrants are in array order (top-left, top-right, bottom-left, bottom-right),
  not asserted to be heliographic east/west.
- `report.json`: source manifest, model revision, local Surya git revision,
  input channel metadata, mask, precision, times, output shapes, memory and status.
  Errors caught by Python leave a failed report; a killed process can leave
  `status: running`, which is not a successful measurement.

Existing output directories are rejected. Large assets and reports live in ignored
`data/experiments/surya_aia`. Downloads use `.part` files; incomplete downloads are
restarted. Run preparation again to reuse completed assets.

Optional `--mode aia-los` retains `hmi_m`; `--mode full` retains all channels for
a reference run. Each mode uses exactly the same normalization and checkpoint.
The loader accepts missing HMI variables in AIA-only mode, rejects missing AIA,
invalid pixels, wrong image size and wrong temporal spacing. It does not silently
resize, impute images, or convert raw FITS into model-ready observations.

## Measured first run — 2026-09-09

The native-resolution AIA-only forward completed successfully using the published
checkpoint at revision `5cc4b5386d5f78fda3896b1389589d4e173bf212`.

| Measurement | Result |
|---|---|
| Host | AMD Ryzen 5 5600; 6 PyTorch CPU threads |
| Precision | float32; no GPU |
| Input | 1×13×2×4096×4096; all five HMI channels masked |
| Encoder output | 1×65536×1280 |
| Single cold forward, validation and pooling | 77.21 seconds |
| Peak process RSS, including loading | 8569.89 MiB (8.37 GiB) |
| Saved pooled vectors | 1×1280 global; 1×5120 spatial |
| Downloaded checkpoint/configs/two frames | 2.77 GiB |

Artifacts are in `data/experiments/surya_aia/probe_cpu/`. This is one sample and
one timed pass: no throughput distribution, GPU measurements, forecast scores,
or evidence of prediction improvement has been obtained. All seven focused tests
passed. The backbone code in `vendor/surya` was left unchanged.

## Next scientific experiment

After the probe succeeds, assemble a larger chronological sample outside the
foundation model's pretraining interval, using identical issue times and targets
for every candidate. Cache embeddings once per issue time. Compare the existing
tabular model with the same model plus embeddings for wind speed at 24/48/72/96 h.
Keep train/validation/test chronological, separate overlapping target windows, fit
any dimensionality reduction on training data only, and account for observation
availability rather than merely image timestamps. Retrospective definitive inputs
do not establish live NRT performance.

Evaluate MAE by horizon, quantile pinball loss and coverage, plus fast-wind events.
A negative frozen-encoder result does not rule out AIA-only adaptation with a head
or LoRA. A small AIA encoder should be a comparison before committing to Surya's
runtime cost. Raw AIA acquisition/calibration and an operational availability-aware
backtest are later work, not delivered by this probe.

## Validation

```bash
.venv/bin/python -m pytest tests/test_surya_aia_probe.py -q
```

Fixtures verify all-five-channel masking by name, normalization, missing AIA,
invalid pixels, temporal alignment, spatial pooling, and strict checkpoint loading
through a tiny real Surya architecture. The tiny random model tests wiring only.
