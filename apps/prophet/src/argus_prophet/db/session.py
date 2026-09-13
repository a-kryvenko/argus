"""Prophet-owned storage; no fallback to administrative or observation roles."""
import os
from sqlalchemy import URL


def get_database_url(*, migration=False):
    key = 'PROPHET_MIGRATION_PASSWORD' if migration else 'PROPHET_DB_PASSWORD'
    missing = [name for name in ('DB_NAME', key) if not os.getenv(name)]
    if missing:
        raise RuntimeError('Missing required database variables: ' + ', '.join(missing))
    return URL.create('postgresql+psycopg',
                      username='argus_prophet_migrator' if migration else 'argus_prophet',
                      password=os.environ[key], database=os.environ['DB_NAME'],
                      host=os.getenv('DB_HOST', 'localhost'), port=int(os.getenv('DB_PORT', '5432')))


def connect():
    import psycopg
    url = get_database_url()
    return psycopg.connect(dbname=url.database, user=url.username, password=url.password,
                           host=url.host, port=url.port, connect_timeout=10,
                           options='-csearch_path=prophet,pg_catalog,pg_temp -cstatement_timeout=60000')
