"""Intelligence-owned SQL connections; never reconnect an active writer."""
import os


def database_url(*, migration=False):
    from sqlalchemy import URL
    key = 'INTELLIGENCE_MIGRATION_PASSWORD' if migration else 'INTELLIGENCE_DB_PASSWORD'
    if not os.getenv('DB_NAME') or not os.getenv(key):
        raise RuntimeError('Missing DB_NAME or ' + key)
    return URL.create('postgresql+psycopg',
                      username='argus_intelligence_migrator' if migration else 'argus_intelligence',
                      password=os.environ[key], database=os.environ['DB_NAME'],
                      host=os.getenv('DB_HOST', 'localhost'), port=int(os.getenv('DB_PORT', '5432')))


def connect():
    import psycopg
    from psycopg.rows import dict_row
    url = database_url()
    return psycopg.connect(dbname=url.database, user=url.username, password=url.password,
                           host=url.host, port=url.port, autocommit=True, row_factory=dict_row,
                           connect_timeout=10, application_name='argus-intelligence',
                           keepalives_idle=30, keepalives_interval=10, keepalives_count=3,
                           tcp_user_timeout=60000,
                           options='-csearch_path=intelligence,pg_catalog,pg_temp -cstatement_timeout=60000')
