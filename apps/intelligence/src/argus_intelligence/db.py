"""Intelligence-owned SQL connections; never reconnect an active writer."""


def database_url():
    """Runtime and Alembic share credentials; passwords remain raw strings."""
    from sqlalchemy import URL
    from common.database import database_parameters
    try:
        parameters = database_parameters('INTELLIGENCE')
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    return URL.create('postgresql+psycopg', **parameters)


def connect():
    import psycopg
    from psycopg.rows import dict_row
    url = database_url()
    return psycopg.connect(url.set(drivername='postgresql').render_as_string(hide_password=False), autocommit=True, row_factory=dict_row,
                           connect_timeout=10, application_name='argus-intelligence',
                           keepalives_idle=30, keepalives_interval=10, keepalives_count=3,
                           tcp_user_timeout=60000,
                           options='-csearch_path=intelligence,pg_catalog,pg_temp -cstatement_timeout=60000')
