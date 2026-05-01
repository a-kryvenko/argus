from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("surya_adapter.solar_wind requires torch") from exc

from common.schema_raw import RawObservationDataPoint


# Public API is intentionally tiny. These constants are implementation policy,
# not user-facing configuration.
AIA_CHANNELS: tuple[str, ...] = (
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "aia1600",
)

HMI_CHANNELS: tuple[str, ...] = (
    "bx",
    "by",
    "bz",
    "dopplergram",
    "magnetogram",
)

LEAD_HOURS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6,
    8, 10, 12, 14, 16, 18, 20, 22, 24,
    27, 30, 33, 36, 39, 42,
    46, 50, 54, 58, 62, 66,
    72, 78, 84, 90, 96,
)

FORECAST_CSV = "solar_wind_forecast.csv"
EMBEDDINGS_NPZ = "embeddings.npz"
MANIFEST_JSON = "input_manifest.json"
IMAGE_SIZE = 512


@dataclass(frozen=True)
class ResolvedImages:
    aia: dict[str, Path]
    hmi: dict[str, Path]


@dataclass(frozen=True)
class ForecastResult:
    output_dir: Path
    forecast_csv: Path
    embeddings_npz: Path | None
    manifest_json: Path
    rows: list[dict[str, Any]]


def forecast(
    observation: RawObservationDataPoint | dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    device: str | torch.device | None = None,
) -> ForecastResult:
    """Forecast solar-wind speed for a sparse 96-hour horizon.

    Parameters
    ----------
    observation:
        RawObservationDataPoint or dict matching that schema.
    output_dir:
        Directory where CSV/NPZ/manifest files will be written.
    device:
        Optional torch device. If omitted, CUDA is used when available.

    Returns
    -------
    ForecastResult
        Paths and in-memory CSV rows.

    Notes
    -----
    This function intentionally exposes no model/config/checkpoint knobs. The
    private Surya backend resolves model assets from the installed/vendored Surya
    package. If your vendored Surya checkout uses different symbols, change only
    _SuryaSolarWindBackend.
    """

    obs = _coerce_observation(observation)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    resolved = _resolve_images(obs, out / "inputs")
    backend = _SuryaSolarWindBackend(device=selected_device)

    rows: list[dict[str, Any]] = []
    lead_embeddings: list[np.ndarray] = []
    image_embedding: np.ndarray | None = None

    issued_at = _to_utc(obs.timestamp)

    for lead_hour in LEAD_HOURS:
        pred = backend.predict(observation=obs, images=resolved, lead_hour=lead_hour)

        target_ts = issued_at + dt.timedelta(hours=lead_hour)
        rows.append(
            {
                "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
                "target_timestamp": target_ts.isoformat().replace("+00:00", "Z"),
                "lead_hours": lead_hour,
                "V": float(pred.wind_speed),
            }
        )

        if pred.image_embedding is not None and image_embedding is None:
            image_embedding = _as_float32_1d(pred.image_embedding)
        if pred.lead_embedding is not None:
            lead_embeddings.append(_as_float32_1d(pred.lead_embedding))

    forecast_csv = out / FORECAST_CSV
    _write_forecast_csv(forecast_csv, rows)

    embeddings_npz = _write_embeddings_npz(
        out / EMBEDDINGS_NPZ,
        issued_at=issued_at,
        image_embedding=image_embedding,
        lead_hours=np.asarray(LEAD_HOURS, dtype=np.int16),
        lead_embeddings=lead_embeddings,
    )

    manifest_json = out / MANIFEST_JSON
    _write_manifest(manifest_json, obs, selected_device, resolved, rows, embeddings_npz)

    return ForecastResult(
        output_dir=out,
        forecast_csv=forecast_csv,
        embeddings_npz=embeddings_npz,
        manifest_json=manifest_json,
        rows=rows,
    )


def _coerce_observation(value: RawObservationDataPoint | dict[str, Any]) -> RawObservationDataPoint:
    if isinstance(value, RawObservationDataPoint):
        return value
    return RawObservationDataPoint.model_validate(value)


def _to_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _resolve_images(observation: RawObservationDataPoint, cache_dir: Path) -> ResolvedImages:
    cache_dir.mkdir(parents=True, exist_ok=True)
    aia = {
        channel: _materialize_image_url(getattr(observation.aia, channel).path, cache_dir / "aia" / channel)
        for channel in AIA_CHANNELS
    }
    hmi = {
        channel: _materialize_image_url(getattr(observation.hmi, channel).path, cache_dir / "hmi" / channel)
        for channel in HMI_CHANNELS
    }
    return ResolvedImages(aia=aia, hmi=hmi)


def _materialize_image_url(url: Any, target_dir: Path) -> Path:
    """Download/cache an HttpUrl image and return a local path.

    Pydantic HttpUrl rejects local file paths, so this adapter treats input paths
    as URLs by design. For local pipelines, expose the files via file server or
    change schema.Image.path to FilePath/AnyUrl in schema.py.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    url_str = str(url)
    parsed = urlparse(url_str)
    ext = Path(parsed.path).suffix or ".fits"
    digest = hashlib.sha256(url_str.encode("utf-8")).hexdigest()[:24]
    target = target_dir / f"{digest}{ext}"
    if target.exists() and target.stat().st_size > 0:
        return target

    tmp = Path(tempfile.mkstemp(prefix=target.name, dir=target_dir)[1])
    try:
        urlretrieve(url_str, tmp)
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _write_forecast_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["issued_at", "target_timestamp", "lead_hours", "V"])
        writer.writeheader()
        writer.writerows(rows)


def _write_embeddings_npz(
    path: Path,
    *,
    issued_at: dt.datetime,
    image_embedding: np.ndarray | None,
    lead_hours: np.ndarray,
    lead_embeddings: list[np.ndarray],
) -> Path | None:
    if image_embedding is None and not lead_embeddings:
        return None

    payload: dict[str, Any] = {
        "issued_at": np.asarray(issued_at.isoformat().replace("+00:00", "Z"), dtype=str),
        "lead_hours": lead_hours,
    }
    if image_embedding is not None:
        payload["image_embedding"] = image_embedding.astype(np.float32, copy=False)
    if lead_embeddings:
        payload["lead_embeddings"] = np.stack(lead_embeddings).astype(np.float32, copy=False)

    np.savez_compressed(path, **payload)
    return path


def _write_manifest(
    path: Path,
    observation: RawObservationDataPoint,
    device: torch.device,
    images: ResolvedImages,
    rows: list[dict[str, Any]],
    embeddings_npz: Path | None,
) -> None:
    manifest = {
        "adapter": "surya_adapter.solar_wind",
        "issued_at": _to_utc(observation.timestamp).isoformat().replace("+00:00", "Z"),
        "device": str(device),
        "lead_hours": list(LEAD_HOURS),
        "forecast_count": len(rows),
        "outputs": {
            "forecast_csv": FORECAST_CSV,
            "embeddings_npz": EMBEDDINGS_NPZ if embeddings_npz is not None else None,
        },
        "inputs": {
            "aia": {k: str(v) for k, v in images.aia.items()},
            "hmi": {k: str(v) for k, v in images.hmi.items()},
            "sensors": observation.sensors.model_dump(),
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _as_float32_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    return arr.reshape(-1)


@dataclass(frozen=True)
class _Prediction:
    wind_speed: float
    image_embedding: np.ndarray | torch.Tensor | None = None
    lead_embedding: np.ndarray | torch.Tensor | None = None


class _SuryaSolarWindBackend:
    """Private integration point for NASA Surya solar-wind downstream model.

    Keep all Surya-package-specific imports and tensor formatting here. The rest
    of the adapter should not know how NASA's package is structured.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

    def predict(
        self,
        *,
        observation: RawObservationDataPoint,
        images: ResolvedImages,
        lead_hour: int,
    ) -> _Prediction:
        batch = self._build_batch(observation=observation, images=images, lead_hour=lead_hour)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            output = self.model(batch)
            wind_speed = self._extract_scalar_prediction(output)

            image_embedding = None
            lead_embedding = None
            try:
                emb = self.model(batch, return_embedding=True)
                image_embedding = emb
            except TypeError:
                # Some Surya heads do not expose this flag. Forecasting should
                # still succeed; embeddings.npz will simply not be written.
                pass

        return _Prediction(wind_speed=wind_speed, image_embedding=image_embedding, lead_embedding=lead_embedding)

    def _load_model(self) -> torch.nn.Module:
        """Load the pretrained Surya solar-wind model.

        This intentionally raises a precise error until wired to vendored
        Surya checkout. In repo, replace this method with the actual loader
        used by downstream_examples/solar_wind_forcasting.
        """

        try:
            # Project-specific hook: create this module if you want zero changes
            # in this adapter file. It must expose load_solar_wind_model().
            from surya_adapter._surya_project_loader import load_solar_wind_model  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Surya solar-wind loader is not connected yet. Add "
                "surya_adapter/_surya_project_loader.py with function "
                "load_solar_wind_model() returning a torch.nn.Module, or replace "
                "_SuryaSolarWindBackend._load_model() with your vendored Surya "
                "downstream loader."
            ) from exc

        model = load_solar_wind_model(device=self.device)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("load_solar_wind_model() must return torch.nn.Module")
        return model

    def _build_batch(
        self,
        *,
        observation: RawObservationDataPoint,
        images: ResolvedImages,
        lead_hour: int,
    ) -> dict[str, torch.Tensor]:
        """Build one Surya solar-wind batch for a requested lead hour.

        This is conservative and explicit. If the vendored Surya dataset exposes
        a canonical single-sample builder, call it here instead of duplicating
        preprocessing.
        """

        try:
            from surya_adapter._surya_project_loader import build_solar_wind_batch  # type: ignore
        except ImportError:
            build_solar_wind_batch = None

        if build_solar_wind_batch is not None:
            batch = build_solar_wind_batch(
                observation=observation,
                aia_paths=images.aia,
                hmi_paths=images.hmi,
                lead_hour=lead_hour,
            )
            if not isinstance(batch, dict):
                raise TypeError("build_solar_wind_batch() must return dict[str, torch.Tensor]")
            return batch

        issued_at = _to_utc(observation.timestamp)
        unix_seconds = int(issued_at.timestamp())

        # Minimal metadata keys matching the downstream inference code shape.
        # Replace/add image tensor keys here according to Surya's expected input.
        return {
            "ts": torch.tensor([[unix_seconds]], dtype=torch.long),
            "time_delta_input": torch.tensor([[0.0]], dtype=torch.float32),
            "forecast": torch.tensor([[float(lead_hour)]], dtype=torch.float32),
            "lead_time_delta": torch.tensor([[float(lead_hour)]], dtype=torch.float32),
            "ds_time": torch.tensor([[unix_seconds + lead_hour * 3600]], dtype=torch.long),
            # Current solar-wind state is useful for downstream models; Surya may
            # ignore this key if its solar-wind head only uses images/time.
            "sensors": torch.tensor(
                [[
                    observation.sensors.Bx,
                    observation.sensors.By,
                    observation.sensors.Bz,
                    observation.sensors.V,
                    observation.sensors.N,
                    observation.sensors.T,
                ]],
                dtype=torch.float32,
            ),
        }

    def _extract_scalar_prediction(self, output: Any) -> float:
        if isinstance(output, torch.Tensor):
            return float(output.detach().cpu().reshape(-1)[0].item())
        if isinstance(output, dict):
            for key in ("prediction", "pred", "V", "wind_speed", "solar_wind_speed"):
                if key in output:
                    return self._extract_scalar_prediction(output[key])
        if isinstance(output, (list, tuple)) and output:
            return self._extract_scalar_prediction(output[0])
        return float(output)

