from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import asyncio
"""Real PostgreSQL storage/retry test; provider reads are fixed deterministic inputs."""
from tests.support.database import database
from datetime import UTC, datetime, timedelta

from .storage import recorder_setup, recorder_database, store_product
from prophet.services.releases.recording import RunRecorder
from prophet.services import verification
from prophet.db.session import get_db_session


async def test_real_releases_are_verified_and_revisions_replace_evidence(recorder_setup, monkeypatch):
    _, _, config, writer = recorder_setup
    issue = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    now = issue + timedelta(hours=3, minutes=30)
    run = (await RunRecorder.begin('solar-wind-speed', 'manual', config, writer=writer))
    from common.schemas.forecast_release import PREDICTION_COLUMNS
    for artifact in ('plasma_speed_quantile', 'plasma_speed_threshold'):
        columns = ['issue_time', 'valid_time', 'lead_hours', *PREDICTION_COLUMNS[artifact]]
        lines = [','.join(columns)]
        for lead in (1, 2, 3):
            lines.append(','.join([issue.isoformat(), (issue + timedelta(hours=lead)).isoformat(), str(lead),
                                   *('0' for _ in PREDICTION_COLUMNS[artifact])]))
        (await run.store(artifact, ('\n'.join(lines) + '\n').encode(), {'sha256': 'test-model'}, 3, columns))
    (await run.finish())
    truth = {'source': 'clio-measurement-hourly-mean-v1', 'status': 'provisional', 'points': [
        {'valid_time': (issue + timedelta(hours=1)).isoformat(), 'metric': 'v', 'value': 400., 'sample_count': 1}]}
    monkeypatch.setattr(verification, 'read_targets', lambda start, end: truth)
    first = (await verification.verify(now=now, writer=writer))
    assert first['products']['solar-wind-speed']['plasma_speed_quantile'] == {
        'total': 3, 'verified': 1, 'missing': 1, 'pending': 1}
    async with get_db_session() as session:
        initial = (await session.scalar(text("SELECT report FROM prophet.forecast_verification WHERE artifact='plasma_speed_quantile'")))
    (await verification.verify(now=now, writer=writer))
    async with get_db_session() as session:
        assert (await session.scalar(text('SELECT count(*) FROM prophet.forecast_verification'))) == 2
        assert (await session.scalar(text("SELECT report FROM prophet.forecast_verification WHERE artifact='plasma_speed_quantile'"))) == initial
    truth['points'][0]['value'] = 450.
    (await verification.verify(now=now, writer=writer))
    async with get_db_session() as session:
        updated = (await session.scalar(text("SELECT report FROM prophet.forecast_verification WHERE artifact='plasma_speed_quantile'")))
    assert updated['evidence_sha256'] != initial['evidence_sha256']
    assert updated['forecast_sha256'] == initial['forecast_sha256']
    report = (await verification.verification_report('solar-wind-speed'))
    quantiles = next(group for group in report['groups'] if group['artifact'] == 'plasma_speed_quantile')
    assert quantiles['tables']['regression.csv'][0]['mae'] == 450.


import pytest

pytestmark = pytest.mark.anyio
