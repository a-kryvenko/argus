from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

IMAGE_SIZE = 512


@dataclass(frozen=True)
class ResolvedImages:
    aia: dict[str, Path]
    hmi: dict[str, Path]


@dataclass(frozen=True)
class ForecastResult:
    rows: list[dict[str, Any]]
    lead_embeddings: list[dict[str, list]]
    embedding: list[float]


def forecast(
    observation: RawObservationDataPoint,
    device: str | None = None,
) -> ForecastResult:
    """Forecast solar-wind speed for a sparse 96-hour horizon.

    Parameters
    ----------
    observation:
        RawObservationDataPoint.
    device:
        Optional torch device. If omitted, CUDA is used when available.

    Returns
    -------
    ForecastResult
    """

    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    backend = _SuryaSolarWindBackend(device=selected_device)

    rows: list[dict[str, Any]] = []
    lead_embeddings: list[np.ndarray] = []
    image_embedding: np.ndarray | None = None

    issued_at = _to_utc(observation.timestamp)

    for lead_hour in LEAD_HOURS:
        pred = backend.predict(observation=observation, images=resolved, lead_hour=lead_hour)

        target_ts = issued_at + dt.timedelta(hours=lead_hour)
        rows.append(
            {
                "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
                "target_timestamp": target_ts.isoformat().replace("+00:00", "Z"),
                "lead_hours": lead_hour,
                "V": float(pred.wind_speed)
            }
        )

        if pred.image_embedding is not None and image_embedding is None:
            image_embedding = _as_float32_1d(pred.image_embedding)
        if pred.lead_embedding is not None:
            lead_embeddings.append({
                "target_timestamp": target_ts.isoformat().replace("+00:00", "Z"),
                "embedding": _as_float32_1d(pred.lead_embedding)
            })

    return ForecastResult(
        embedding=image_embedding,
        rows=rows,
        lead_embeddings=lead_embeddings
    )

def _to_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)

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
        observation: RawObservationDataPoint,
        images: ResolvedImages,
        lead_hour: int,
    ) -> _Prediction:
        batch = self._build_batch(observation=observation, images=images, lead_hour=lead_hour)

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
        # Initialize model
        model = get_model(config, wandb_logger=None)
        
        # Apply LoRA if needed
        if config["model"]["use_lora"]: 
            model = apply_peft_lora(model, config) 
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
        
        # Remove 'module.' prefix if present (from DistributedDataParallel)
        if any(key.startswith('module.') for key in model_state.keys()):
            model_state = {key.replace('module.', ''): value for key, value in model_state.items()}
        
        # Load state dict
        try:
            model.load_state_dict(model_state, strict=True)
        except Exception as e:
            print(f"Failed to load with strict=True: {e}")
            raise e
        
        model.to(device)
        model.eval()
        
        return model

    def _build_batch(
        self,
        *,
        observation: RawObservationDataPoint,
        lead_hour: int,
    ) -> dict[str, torch.Tensor]:
        batch = None

        issued_at = _to_utc(observation.timestamp)
        unix_seconds = int(issued_at.timestamp())

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

