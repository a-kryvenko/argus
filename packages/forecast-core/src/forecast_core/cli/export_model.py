from __future__ import annotations

import argparse
from pathlib import Path
import torch
from forecast_core.models.forecast_model import ForecastModel
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    model = ForecastModel(config)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    torch.save({"model": model.state_dict(), "config": config}, args.output)


if __name__ == "__main__":
    main()
