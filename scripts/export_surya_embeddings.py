#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

from surya_adapter.embeddings.exporter import export_embeddings


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'surya' / 'export_embeddings.yaml'


def _resolve(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _load_vendor_module(module_name: str, file_path: Path):
    module_dir = str(file_path.parent)

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module from {file_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main() -> None:
    parser = argparse.ArgumentParser(description='Export pooled Surya embeddings to NPZ.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='Project config YAML')
    parser.add_argument('--device', default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    args = parser.parse_args()

    project_cfg_path = args.config.resolve()
    project_cfg = yaml.safe_load(project_cfg_path.read_text())
    base_dir = REPO_ROOT

    vendor_root = _resolve(base_dir, project_cfg['surya']['vendor_root'])
    config_path = _resolve(base_dir, project_cfg['surya']['config_path'])
    checkpoint_path = _resolve(base_dir, project_cfg['surya']['checkpoint_path'])
    output_path = _resolve(base_dir, project_cfg['surya']['output_path'])
    example_subdir = project_cfg.get('example_subdir')
    device = args.device or project_cfg.get('device', 'cpu')
    num_samples = args.num_samples if args.num_samples is not None else project_cfg.get('num_samples')

    example_dir = _resolve(vendor_root, 'downstream_examples/solar_wind_forcasting')

    dataset_mod = _load_vendor_module(
        'argus_vendor_surya_dataset',
        example_dir / 'dataset.py',
    )
    finetune_mod = _load_vendor_module(
        'argus_vendor_surya_finetune_for_export',
        example_dir / 'finetune.py',
    )
    utils_data_mod = _load_vendor_module(
        'argus_vendor_surya_utils_data_for_export',
        vendor_root / 'surya' / 'utils' / 'data.py',
    )

    surya_cfg = yaml.safe_load(config_path.read_text())
    config_dir = config_path.parent

    def resolve_surya_config_path(value: str | None):
        if value is None:
            return None
        p = Path(value)
        if not p.is_absolute():
            p = (config_dir / p).resolve()
        return str(p)

    surya_cfg['data']['scalers_path'] = resolve_surya_config_path(surya_cfg['data'].get('scalers_path'))
    surya_cfg['data']['train_data_path'] = resolve_surya_config_path(surya_cfg['data'].get('train_data_path'))
    surya_cfg['data']['solarwind_train_index'] = resolve_surya_config_path(surya_cfg['data'].get('solarwind_train_index'))
    surya_cfg['data']['sdo_data_root_path'] = resolve_surya_config_path(surya_cfg['data'].get('sdo_data_root_path'))
    surya_cfg['pretrained_path'] = resolve_surya_config_path(surya_cfg.get('pretrained_path'))
    surya_cfg['data']['scalers'] = yaml.safe_load(Path(surya_cfg['data']['scalers_path']).read_text())

    def dataset_factory(cfg):
        scalers = utils_data_mod.build_scalers(info=cfg['data']['scalers'])
        return dataset_mod.WindSpeedDSDataset(
            sdo_data_root_path=cfg['data']['sdo_data_root_path'],
            index_path=cfg['data']['train_data_path'],
            time_delta_input_minutes=cfg['data']['time_delta_input_minutes'],
            time_delta_target_minutes=cfg['data']['time_delta_target_minutes'],
            n_input_timestamps=cfg['model']['time_embedding']['time_dim'],
            rollout_steps=cfg['rollout_steps'],
            channels=cfg['data']['channels'],
            drop_hmi_probability=cfg['drop_hmi_probability'],
            num_mask_aia_channels=cfg['num_mask_aia_channels'],
            use_latitude_in_learned_flow=cfg['use_latitude_in_learned_flow'],
            scalers=scalers,
            phase='train',
            ds_solar_wind_path=cfg['data']['solarwind_train_index'],
            ds_time_column=cfg['data']['ds_time_column'],
            ds_time_delta_in_out=cfg['data']['ds_time_delta_in_out'],
            ds_time_tolerance=cfg['data']['ds_time_tolerance'],
            ds_match_direction=cfg['data']['ds_match_direction'],
            ds_normalize=cfg['data']['ds_normalize'],
            ds_scaler=cfg['data']['ds_scaler'],
        )

    out = export_embeddings(
        vendor_root=vendor_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        dataset_factory=dataset_factory,
        collate_fn=finetune_mod.custom_collate_fn,
        num_samples=num_samples,
        device=device,
        example_subdir=example_subdir,
        project_config=project_cfg,   # <- pass Argus config through
    )
    print(f'saved embeddings to {out}')


if __name__ == '__main__':
    main()
