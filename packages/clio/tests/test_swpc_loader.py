from datetime import UTC, datetime

import pandas as pd

from clio.dataloaders.swpc_loader import SWPC_Loader


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
        staticmethod(lambda: (sensor_frame, index_frame)),
    )

    result = SWPC_Loader.load_measurements()

    assert set(result["metric"]) == {"bx", "by", "kp"}
    assert set(result["observed_at"]) == {first_time, second_time}
    assert len(result) == 5
