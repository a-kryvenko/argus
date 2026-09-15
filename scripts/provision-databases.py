"""Create one database and owner per domain; never move data or rotate passwords.

Without --apply, inspect the plan without writing. CREATE DATABASE cannot run in
a transaction: a failed apply may leave completed databases; fix and rerun.
"""
import argparse
import os
from pathlib import Path

import psycopg
from psycopg import sql
from sqlalchemy import URL
from sqlalchemy.engine import make_url
from common.database import database_parameters

DOMAINS = ('api', 'clio', 'prophet', 'intelligence')


def connection_url(value):
    try:
        url = make_url(value)
        if url.get_backend_name() != 'postgresql' or not all((url.host, url.database, url.username, url.password)):
            raise ValueError()
        return url.set(drivername='postgresql')
    except Exception:
        raise ValueError('Expected a PostgreSQL URL with host, database, user and password') from None


def provision(admin_url, databases, *, apply=False):
    """Provision only the supplied databases on the administrator's server."""
    admin_url = connection_url(admin_url)
    targets = {domain: connection_url(value) for domain, value in databases.items()}
    if len({url.database for url in targets.values()}) != len(targets):
        raise ValueError('Each domain must have a separate database')
    if len({url.username for url in targets.values()}) != len(targets):
        raise ValueError('Each domain must have a separate owner')
    for url in targets.values():
        if (url.host, url.port or 5432) != (admin_url.host, admin_url.port or 5432):
            raise ValueError('All databases must be on the administrative connection server')
        if url.database in ('postgres', 'template0', 'template1', admin_url.database) or url.username == admin_url.username:
            raise ValueError('Service databases and owners must be separate from administration')
    with psycopg.connect(admin_url.render_as_string(hide_password=False), autocommit=True) as admin:
        admin.execute('SELECT pg_advisory_lock(730999)')
        # Validate every existing target before creating anything.
        existing = {}
        for domain, url in targets.items():
            owner = admin.execute('SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname=%s', (url.database,)).fetchone()
            if owner and owner[0] != url.username:
                raise ValueError(f'{domain}: existing database has a different owner')
            role = admin.execute('SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls FROM pg_roles WHERE rolname=%s', (url.username,)).fetchone()
            if role and any(role):
                raise ValueError(f'{domain}: service owner has administrative privileges')
            if role and owner:
                # Existing passwords are verified, never silently changed.
                with psycopg.connect(url.render_as_string(hide_password=False)) as conn:
                    conn.execute('SELECT 1')
            existing[domain] = (bool(role), bool(owner))
        for domain, url in targets.items():
            role_exists, database_exists = existing[domain]
            print(f'{domain}: {url.database}, owner {url.username}; ' +
                  ('exists' if database_exists else 'create database'))
            if not apply:
                continue
            if not role_exists:
                admin.execute(sql.SQL('CREATE ROLE {} LOGIN PASSWORD {}').format(
                    sql.Identifier(url.username), sql.Literal(url.password)))
            if not database_exists:
                admin.execute(sql.SQL('CREATE DATABASE {} OWNER {}').format(
                    sql.Identifier(url.database), sql.Identifier(url.username)))
            # Ownership grants access to this database. No per-object ACL matrix.
            admin.execute(sql.SQL('REVOKE ALL ON DATABASE {} FROM PUBLIC').format(sql.Identifier(url.database)))
            with psycopg.connect(url.render_as_string(hide_password=False)) as conn:
                conn.execute('SELECT 1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from dotenv import load_dotenv
    root = Path(os.getenv('ARGUS_WORKDIR', Path(__file__).resolve().parents[1]))
    load_dotenv(root / '.env')
    load_dotenv(root / '.env.local', override=True)
    try:
        provision(URL.create('postgresql', **database_parameters('')),
                  {d: URL.create('postgresql', **database_parameters(d.upper())) for d in DOMAINS},
                  apply=args.apply)
    except ValueError as exc:
        # These are our fixed validation messages, never the original URL parser error.
        parser.exit(1, str(exc) + '\n')
    except psycopg.Error:
        # Driver errors may contain connection credentials.
        parser.exit(1, 'Database provisioning failed: verify DB settings, existing owners/passwords and administrator access. No passwords were rotated.\n')
    print('Applied' if args.apply else 'Plan only; use --apply to create databases')


if __name__ == '__main__':
    main()
