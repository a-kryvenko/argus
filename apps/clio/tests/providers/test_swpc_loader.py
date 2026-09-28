from datetime import UTC, datetime

import pandas as pd

from clio.providers.swpc_loader import SWPC_Loader


def test_load_measurements_preserves_source_timestamps(monkeypatch) -> None:
    first_time = datetime(2026, 8, 30, 0, 7, tzinfo=UTC)
    second_time = datetime(2026, 8, 30, 0, 52, tzinfo=UTC)
    sensor_frame = pd.DataFrame({
        "issue_time": [first_time, second_time],
        "bx": [1.0, 2.0],
        "by": [3.0, 4.0],
    })
    index_frame = pd.DataFrame({
        "issue_time": [first_time],
        "kp": [5.0],
    })
    monkeypatch.setattr(
        SWPC_Loader,
        "_fetch_source_frames",
        staticmethod(lambda **kwargs: (sensor_frame, index_frame)),
    )

    result = SWPC_Loader.load_measurements()

    assert set(result["metric"]) == {"bx", "by", "kp"}
    assert set(result["observed_at"]) == {first_time, second_time}
    assert len(result) == 5


def test_model_refresh_skips_kp_download(monkeypatch):
    from unittest.mock import Mock
    empty = pd.DataFrame(columns=['issue_time'])
    for name in ('_fetch_live_sensors', '_fetch_f10_7_flux', '_fetch_dst'):
        monkeypatch.setattr(SWPC_Loader, name, Mock(return_value=empty))
    kp = Mock(return_value=empty)
    monkeypatch.setattr(SWPC_Loader, '_fetch_live_kp', kp)
    SWPC_Loader.load_measurements(include_kp=False)
    kp.assert_not_called()
    SWPC_Loader.load_measurements()
    kp.assert_called_once()
