from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forecast_core.data_pipelines.omni_dscovr_sdo.scalers import save_scalers


def compute_scalers_from_dataset(dataset: dict[str, np.ndarray], output_path: str | Path) -> dict[str, list[float]]:
    history = dataset['history'].reshape(-1, dataset['history'].shape[-1])
    context = dataset['context']
    target = dataset['targets'].reshape(-1, dataset['targets'].shape[-1])
    payload = {
        'history_mean': history.mean(axis=0).tolist(),
        'history_std': (history.std(axis=0) + 1e-6).tolist(),
        'context_mean': context.mean(axis=0).tolist(),
        'context_std': (context.std(axis=0) + 1e-6).tolist(),
        'target_mean': target.mean(axis=0).tolist(),
        'target_std': (target.std(axis=0) + 1e-6).tolist(),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_scalers(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    compute_scalers_from_dataset(data, args.output)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
