from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import yaml


def _assert_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    
def _load_python_module(module_name: str, file_path: Path) -> ModuleType:
    module_dir = str(file_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_example_dir(vendor_root: Path, example_subdir: str | None = None) -> Path:
    candidates = []
    if example_subdir:
        candidates.append(vendor_root / example_subdir)
    candidates.extend([
        vendor_root / 'downstream_examples' / 'solar_wind_forcasting',
        vendor_root / 'downstream_examples' / 'solar_wind_forecasting',
    ])
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError('Could not locate vendor Surya solar wind example directory')


def load_surya_model(
    vendor_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    example_subdir: str | None = None,
    project_config: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    vendor_root = Path(vendor_root).resolve()
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    _assert_exists(vendor_root, "vendor_root")
    _assert_exists(config_path, "config_path")
    _assert_exists(checkpoint_path, "checkpoint_path")

    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    config = yaml.safe_load(config_path.read_text())
    config_dir = config_path.parent

    example_dir = _resolve_example_dir(vendor_root, example_subdir=example_subdir)

    def _resolve_cfg_path(value: str | None) -> str | None:
        if not value:
            return None
        p = Path(value)
        if not p.is_absolute():
            p = (config_dir / p).resolve()
        return str(p)

    # First resolve vendor-config-relative paths
    if "data" in config:
        config["data"]["scalers_path"] = _resolve_cfg_path(config["data"].get("scalers_path"))
        config["data"]["train_data_path"] = _resolve_cfg_path(config["data"].get("train_data_path"))
        config["data"]["solarwind_train_index"] = _resolve_cfg_path(config["data"].get("solarwind_train_index"))
        config["data"]["sdo_data_root_path"] = _resolve_cfg_path(config["data"].get("sdo_data_root_path"))

    config["pretrained_path"] = _resolve_cfg_path(config.get("pretrained_path"))

    # Then override from Argus project config if provided
    if project_config is not None:
        surya_cfg = project_config.get("surya", {})

        if surya_cfg.get("pretrained_path"):
            config["pretrained_path"] = str(Path(surya_cfg["pretrained_path"]).resolve())

        if "data" in config and surya_cfg.get("scalers_path"):
            config["data"]["scalers_path"] = str(Path(surya_cfg["scalers_path"]).resolve())

        if "data" in config and surya_cfg.get("sdo_data_root_path"):
            config["data"]["sdo_data_root_path"] = str(Path(surya_cfg["sdo_data_root_path"]).resolve())

        if "data" in config and surya_cfg.get("train_data_path"):
            config["data"]["train_data_path"] = str(Path(surya_cfg["train_data_path"]).resolve())

        if "data" in config and surya_cfg.get("solarwind_train_index"):
            config["data"]["solarwind_train_index"] = str(Path(surya_cfg["solarwind_train_index"]).resolve())

    # Validate critical files after override
    if config.get("pretrained_path"):
        _assert_exists(Path(config["pretrained_path"]), "pretrained_path")

    if "data" in config and config["data"].get("scalers_path"):
        _assert_exists(Path(config["data"]["scalers_path"]), "scalers_path")
        config["data"]["scalers"] = yaml.safe_load(Path(config["data"]["scalers_path"]).read_text())

    infer_module = _load_python_module(
        "argus_vendor_surya_infer",
        example_dir / "infer.py",
    )

    model = infer_module.load_model(
        config=config,
        checkpoint_path=str(checkpoint_path),
        device=device,
    )
    model.to(device)
    model.eval()
    return model, config