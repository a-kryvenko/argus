import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from common.schemas.forecast_release import ForecastArtifact, ForecastRelease


def artifact(name='dst_quantile', value='-10'):
    text = 'issue_time,valid_time,lead_hours,dst_q10,dst_q50,dst_q90\n2026-09-13T00:00:00Z,2026-09-13T01:00:00Z,1,-20,' + value + ',0\n'
    return ForecastArtifact(name=name, sha256=hashlib.sha256(text.encode()).hexdigest(),
                            csv_text=text, row_count=1, columns=text.splitlines()[0].split(','), model_info={})


def release():
    return ForecastRelease(release_id=uuid4(), run_id=uuid4(), product='dst',
                           published_at=datetime.now(UTC), issue_time=datetime(2026, 9, 13, tzinfo=UTC),
                           artifacts=[artifact()])


def test_rejects_corrupt_and_incomplete_releases():
    payload = release().model_dump()
    payload['artifacts'][0]['csv_text'] += 'corrupt'
    with pytest.raises(ValueError, match='checksum'):
        ForecastRelease.model_validate(payload)
    payload = release().model_dump()
    payload['product'] = 'solar-wind-speed'
    with pytest.raises(ValueError, match='Incomplete'):
        ForecastRelease.model_validate(payload)
    payload = release().model_dump()
    payload['issue_time'] = datetime(2026, 9, 12, tzinfo=UTC)
    with pytest.raises(ValueError, match='Mixed'):
        ForecastRelease.model_validate(payload)


@pytest.mark.parametrize('value', ['nan', 'inf', 'not-a-number'])
def test_nonfinite_or_invalid_predictions_cannot_be_published(value):
    with pytest.raises(ValueError):
        artifact(value=value)


def test_missing_prediction_columns_are_rejected():
    payload = artifact().model_dump()
    text = 'issue_time,valid_time,lead_hours\n2026-09-13T00:00:00Z,2026-09-13T01:00:00Z,1\n'
    payload.update(csv_text=text, columns=text.splitlines()[0].split(','), sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError, match='prediction columns'):
        ForecastArtifact.model_validate(payload)
