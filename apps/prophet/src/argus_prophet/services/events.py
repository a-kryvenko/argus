"""Compatibility boundary; implementation lives in private forecast-core."""
from forecast_core.api import (
    detect_l1_compressions as l1_detections,
)
from argus_prophet.services.cme_arrival import forecast_events as forecast

def calculate(inputs, *, speed_calculator=None, registry=None):
    """Serialize the private forecast using the standard Prophet release ledger."""
    import pandas as pd
    from common.schemas.solar_wind_events import CMESpeedOutlook
    from argus_prophet.services.generation.calculation import Artifact
    report = forecast(inputs, registry=registry, include_hss=False, include_dbem=False)
    report.model_version = 'solar-wind-events-v4'
    if speed_calculator is not None:
        report.cme_speed = CMESpeedOutlook.model_validate(speed_calculator(report.events))
    else:
        report.cme_speed = CMESpeedOutlook(model_version='cme-speed-v2', state='unavailable', reason='Speed runtime not configured')
    report.limitations = [
        'Speed scenarios are conditional on the ENLIL predicted impact; scenario bounds are not calibrated quantiles or hit probabilities.',
        'The speed model was trained through2024 and calibrated/selected on2025;2026 is evaluation only.',
        'Multi-CME runs have no validated speed head; excluded events remain visible in the arrival forecast.',
        'L1 compression candidates do not confirm CME attribution or automatically reanchor the speed scenario.',
        'Bz and storm strength are not predicted. HSS and DBEM forecasts are disabled for this release.',
    ]
    frame = pd.DataFrame([dict(issue_time=report.issue_time, valid_time=report.issue_time,
        lead_hours=0, event_count=len(report.events), detection_count=len(report.l1_detections),
        hss_count=0, cme_ensemble_count=0, report_json=report.model_dump_json())])
    return [Artifact('solar_wind_events', frame.to_csv(index=False).encode(),
        {'model': report.model_version, 'registry_name': 'solar_wind_events', 'experimental': True,
         'input_cutoff': inputs.as_of.isoformat(), 'source_status': report.source_status,
         'cme_arrival': report.source_status['CME_arrival'],
         'cme_speed': {'state': report.cme_speed.state, 'model_sha256': report.cme_speed.model_sha256}},
        1, list(frame.columns))]
