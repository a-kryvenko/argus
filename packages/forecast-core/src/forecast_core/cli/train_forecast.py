from __future__ import annotations

import argparse
from forecast_core.training.trainer import ForecastTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    ForecastTrainer(args.config).fit()


if __name__ == "__main__":
    main()
