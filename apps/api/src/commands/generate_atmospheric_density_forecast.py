from datetime import datetime, timezone

from common.config import get_config
from forecast_core.inference.AtmosphericDensityForecastService import (
    AtmosphericDensityForecastService,
)


def main() -> None:
    config = get_config()
    registry = config.models_registry["models"][AtmosphericDensityForecastService.registry_name]
    output_path = config.workdir / registry["forecast_path"]
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    service = AtmosphericDensityForecastService()
    frame = service.forecast_grid()
    issue_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    frame.insert(0, "issue_time", issue_time)
    frame.insert(
        2,
        "lead_hours",
        ((frame["valid_time"] - issue_time).dt.total_seconds() / 3600).astype(int),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    print(f"Saved {len(frame):,} JB2008 density rows to {output_path}")


if __name__ == "__main__":
    main()
