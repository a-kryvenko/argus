"""Administrative domain provisioning. Credentials exist only in the maintenance container."""
import argparse
import os

import psycopg
from psycopg import sql

DOMAIN = 'intelligence'
ROLES = ('argus_intelligence', 'argus_intelligence_migrator')


def provision(conn, passwords, *, rotate=False):
    conn.execute('SELECT pg_advisory_xact_lock(730999)')
    current, database_owner = conn.execute('SELECT current_user, pg_get_userbyid(datdba) FROM pg_database WHERE datname=current_database()').fetchone()
    if current in ROLES or database_owner in ROLES:
        raise RuntimeError('Intelligence roles cannot own or administer the database')
    owner = conn.execute("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname='intelligence'").fetchone()
    if owner and owner[0] != ROLES[1]:
        raise RuntimeError('Unexpected Intelligence schema owner')
    for role, key in zip(ROLES, ('INTELLIGENCE_DB_PASSWORD', 'INTELLIGENCE_MIGRATION_PASSWORD')):
        existing = conn.execute('SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls,rolinherit FROM pg_roles WHERE rolname=%s', (role,)).fetchone()
        if existing and (any(existing) or conn.execute('SELECT 1 FROM pg_auth_members WHERE member=(SELECT oid FROM pg_roles WHERE rolname=%s) OR roleid=(SELECT oid FROM pg_roles WHERE rolname=%s)', (role, role)).fetchone()):
            raise RuntimeError('Unexpected Intelligence role privileges')
        if not existing:
            conn.execute(sql.SQL('CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD {}')
                         .format(sql.Identifier(role), sql.Literal(passwords[key])))
        elif rotate:
            conn.execute(sql.SQL('ALTER ROLE {} PASSWORD {}').format(sql.Identifier(role), sql.Literal(passwords[key])))
    if not owner:
        conn.execute('CREATE SCHEMA intelligence AUTHORIZATION argus_intelligence_migrator')
        conn.execute('REVOKE ALL ON SCHEMA intelligence FROM PUBLIC')
        conn.execute('GRANT USAGE ON SCHEMA intelligence TO argus_intelligence')
        conn.execute('ALTER DEFAULT PRIVILEGES FOR ROLE argus_intelligence_migrator IN SCHEMA intelligence GRANT SELECT,INSERT,UPDATE,DELETE ON TABLES TO argus_intelligence')
        conn.execute('ALTER DEFAULT PRIVILEGES FOR ROLE argus_intelligence_migrator REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rotate-passwords', action='store_true')
    args = parser.parse_args()
    required = ('DB_NAME', 'DB_USER', 'DB_PASSWORD', 'INTELLIGENCE_DB_PASSWORD', 'INTELLIGENCE_MIGRATION_PASSWORD')
    if any(not os.getenv(key) for key in required):
        raise SystemExit('Missing administrative Intelligence provisioning variables')
    connection = dict(dbname=os.environ['DB_NAME'], host=os.getenv('DB_HOST', 'localhost'),
                      port=int(os.getenv('DB_PORT', '5432')), connect_timeout=10)
    try:
        with psycopg.connect(**connection, user=os.environ['DB_USER'], password=os.environ['DB_PASSWORD']) as conn:
            provision(conn, os.environ, rotate=args.rotate_passwords)
        # Validate supplied credentials without rotating existing passwords during ordinary deploys.
        for role, key in zip(ROLES, ('INTELLIGENCE_DB_PASSWORD', 'INTELLIGENCE_MIGRATION_PASSWORD')):
            with psycopg.connect(**connection, user=role, password=os.environ[key]) as conn:
                conn.execute('SELECT 1')
    except Exception:
        raise SystemExit('Intelligence provisioning or credential verification failed') from None
    print('Intelligence domain provisioned and credentials verified')
