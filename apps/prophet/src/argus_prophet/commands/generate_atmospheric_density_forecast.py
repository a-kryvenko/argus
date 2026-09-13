import logging
from time import perf_counter

from common.schemas.forecast_inputs import ForecastInputs
from argus_prophet.observations import load_inputs
from argus_prophet.services.density_observations import load_density_drivers
from common.config import get_config
from forecast_core.api import (
    AtmosphericDensityForecastService,
)


def main(inputs: ForecastInputs | None = None, recorder=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = perf_counter()
    print("JB2008: loading internal observations and history", flush=True)
    config = get_config()
    registry = config.models_registry["models"][
        AtmosphericDensityForecastService.registry_name
    ]
    output_path = config.workdir / registry["forecast_path"]
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    service = AtmosphericDensityForecastService()
    inputs = inputs if inputs is not None else load_inputs()
    issue_time = inputs.as_of.replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    drivers = load_density_drivers(inputs, issue_time)
    print(f"JB2008: drivers ready in {perf_counter() - started:.1f}s; "
          f"calculating {len(drivers)} hourly grids (first grid includes JIT compilation)", flush=True)
    frame = service.forecast_grid(
        drivers=drivers,
        progress=lambda done, total: print(
            f"JB2008: grid {done}/{total}, elapsed {perf_counter() - started:.1f}s", flush=True),
    )
    frame.insert(0, "issue_time", issue_time)
    frame.insert(
        2,
        "lead_hours",
        ((frame["valid_time"] - issue_time).dt.total_seconds() / 3600).astype(int),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"JB2008: saving {len(frame):,} rows", flush=True)
    frame.to_csv(temporary_path, index=False)
    if recorder:
        recorder.store(service.registry_name, temporary_path,
                       {"backend": "forecast_core", "registry_name": service.registry_name,
                        "issue_time": issue_time.isoformat()}, len(frame), list(frame.columns))
    temporary_path.replace(output_path)
    if recorder:
        recorder.csv_written(service.registry_name)
    print(f"Saved {len(frame):,} JB2008 density rows to {output_path} "
          f"in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
