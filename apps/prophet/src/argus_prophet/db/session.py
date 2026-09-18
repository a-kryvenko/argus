"""Prophet storage; writers reuse the connection holding their advisory lock."""
from contextlib import contextmanager
from contextvars import ContextVar


_writer_session = ContextVar('prophet_writer_session', default=None)


def get_database_url():
    """Runtime and Alembic share credentials; passwords remain raw strings."""
    from sqlalchemy import URL
    from common.database import database_parameters
    try:
        parameters = database_parameters('PROPHET')
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    return URL.create('postgresql+psycopg', **parameters)


def open_connection(*, autocommit=False):
    import psycopg
    url = get_database_url()
    return psycopg.connect(url.set(drivername='postgresql').render_as_string(hide_password=False), connect_timeout=10,
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
