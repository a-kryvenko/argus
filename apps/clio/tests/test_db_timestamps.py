from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select

from clio.db.models.file import File


def test_test_database_preserves_timezone_aware_instants(storage):
    timestamp = datetime(2026, 11, 4, 12, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    expected = timestamp.astimezone(UTC)
    with storage.engine.begin() as connection:
        connection.execute(File.__table__.insert().values(
            id=uuid4(), timestamp=timestamp, bucket='argus-private', key='example'))
        row = connection.execute(select(File.timestamp.label('file_timestamp'))
            .where(File.timestamp == expected)).mappings().one()
    assert row['file_timestamp'].tzinfo is not None
    assert row['file_timestamp'] == expected
