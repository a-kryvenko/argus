from datetime import UTC, datetime
from unittest.mock import Mock

from common.schemas.forecast_inputs import AIAFeatureFrame, ForecastInputs, SpeedObservation
from common.schemas.observation import Observation
from forecast.api import SWSpeedFS
from .test_aia_wind_service import bundle


def test_snapshot_preserves_unfilled_speed_and_aia_receipts():
    now = datetime(2026, 9, 29, tzinfo=UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]),
        speed_observations=[SpeedObservation(issue_time=now, v=399)],
        aia_frames=[AIAFeatureFrame(slot_at=now, observed_at=now, available_at=now,
                                   sha256='a'*64, features={'aia_area_sector': .2})])
    # Input preparation must not fall back to normalized observations or files.
    service = SWSpeedFS(bundle())
    service.forecast = Mock(return_value=object())
    import common.adapters
    from unittest.mock import patch
    with patch.object(common.adapters, 'forecast_to_dataframe', side_effect=lambda x: x):
        service.forecast_snapshot(inputs, issue_time=now)
    options = service.forecast.call_args.kwargs
    assert options['speed_history'].v.tolist() == [399]
    assert options['speed_history'].issue_time.tolist() == [now]
    assert options['aia_features'].aia_area_sector.tolist() == [.2]
    assert options['aia_features'].available_at.tolist() == [now]
