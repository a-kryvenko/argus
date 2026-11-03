"""Registry of successfully uploaded immutable objects."""
from functools import lru_cache
from datetime import UTC

from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert

from common.db.session import get_database_url
from clio.db.models.file import File


@lru_cache(maxsize=1)
def engine():
    return create_engine(get_database_url('CLIO'), pool_pre_ping=True,
                         connect_args={'options': '-csearch_path=clio,public,pg_catalog,pg_temp'})


def register(*, timestamp, bucket, key, connection=None):
    if timestamp.tzinfo is None:
        raise ValueError('File timestamp must include a timezone')
    statement = insert(File).values(timestamp=timestamp.astimezone(UTC), bucket=bucket, key=key
        ).on_conflict_do_nothing(index_elements=['bucket', 'key'])
    if connection is None:
        with engine().begin() as connection:
            return register(timestamp=timestamp, bucket=bucket, key=key, connection=connection)
    connection.execute(statement)
    record = connection.execute(select(File.__table__).where(File.bucket == bucket, File.key == key)).mappings().one()
    return normalize(dict(record))


def normalize(record):
    if record['timestamp'].tzinfo is None:
        record['timestamp'] = record['timestamp'].replace(tzinfo=UTC)
    return record


def find(*, id=None, bucket=None, key=None, start=None, end=None, as_of=None, limit=10000):
    statement = select(File.__table__)
    for name, value in dict(id=id, bucket=bucket, key=key).items():
        if value is not None:
            statement = statement.where(getattr(File, name) == value)
    if start is not None:
        statement = statement.where(File.timestamp >= start)
    if end is not None:
        statement = statement.where(File.timestamp < end)
    if as_of is not None:
        statement = statement.where(File.timestamp <= as_of)
    with engine().connect() as connection:
        rows = connection.execute(statement.order_by(File.timestamp.desc(), File.key).limit(limit)).mappings().all()
    return [normalize(dict(row)) for row in rows]


def dispose():
    if engine.cache_info().currsize:
        engine().dispose()
        engine.cache_clear()
