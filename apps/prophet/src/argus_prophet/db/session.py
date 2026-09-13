"""Prophet storage; writers reuse the connection holding their advisory lock."""
import os
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import URL

_writer_session = ContextVar('prophet_writer_session', default=None)


def get_database_url(*, migration=False):
    key = 'PROPHET_MIGRATION_PASSWORD' if migration else 'PROPHET_DB_PASSWORD'
    missing = [name for name in ('DB_NAME', key) if not os.getenv(name)]
    if missing:
        raise RuntimeError('Missing required database variables: ' + ', '.join(missing))
    return URL.create('postgresql+psycopg',
                      username='argus_prophet_migrator' if migration else 'argus_prophet',
                      password=os.environ[key], database=os.environ['DB_NAME'],
                      host=os.getenv('DB_HOST', 'localhost'), port=int(os.getenv('DB_PORT', '5432')))


def open_connection(*, autocommit=False):
    import psycopg
    url = get_database_url()
    return psycopg.connect(dbname=url.database, user=url.username, password=url.password,
                           host=url.host, port=url.port, connect_timeout=10,
                           autocommit=autocommit, application_name='argus-prophet',
                           keepalives_idle=30, keepalives_interval=10, keepalives_count=3,
                           tcp_user_timeout=60000,
                           options='-csearch_path=prophet,pg_catalog,pg_temp -cstatement_timeout=60000')


def require_writer():
    conn = _writer_session.get()
    if conn is None:
        raise RuntimeError('Prophet writes require the database generation lock')
    if conn.closed or conn.broken:
        raise RuntimeError('Prophet lock connection was lost; reacquire the lock for a new attempt')
    return conn


def check_writer():
    # Detect server-side session loss before external file replacement, too.
    require_writer().execute('SELECT 1')


@contextmanager
def writer_session(conn):
    if _writer_session.get() is not None:
        raise RuntimeError('Prophet generation lock is not reentrant')
    token = _writer_session.set(conn)
    try:
        yield
    finally:
        _writer_session.reset(token)


@contextmanager
def connect(*, writing=False):
    conn = require_writer() if writing else _writer_session.get()
    if conn is not None:
        # Never silently reconnect a writer: a new connection would not own its lock.
        with conn.transaction():
            yield conn
    else:
        with open_connection() as connection:
            yield connection
