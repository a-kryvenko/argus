"""Prophet storage; writers reuse the connection holding their advisory lock."""
from contextlib import contextmanager


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


@contextmanager
def transaction(writer):
    """Use the exact lock-owning connection; never reconnect a failed writer."""
    if writer is None:
        raise RuntimeError('Prophet writes require a database writer lock')
    if writer.closed or writer.broken:
        raise RuntimeError('Prophet lock connection was lost; reacquire the lock for a new attempt')
    with writer.transaction():
        yield writer


@contextmanager
def connect():
    """Independent read connection; writers pass their connection explicitly."""
    with open_connection() as connection:
        yield connection
