from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import yaml


def stats(arr):
    return arr.mean(axis=0).tolist(), (arr.std(axis=0) + 1e-6).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.train_npz)
    history = data["history"].reshape(-1, data["history"].shape[-1])
    context = data["context"]
    targets = data["targets"].reshape(-1, data["targets"].shape[-1])
    payload = {
        "history_mean": stats(history)[0],
        "history_std": stats(history)[1],
        "context_mean": stats(context)[0],
        "context_std": stats(context)[1],
        "target_mean": stats(targets)[0],
        "target_std": stats(targets)[1],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
