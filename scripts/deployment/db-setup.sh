#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

apply=false

case "$#" in
    0) ;;
    1)
        if [[ "$1" != "--apply" ]]; then
            echo "Usage: $0 [--apply]" >&2
            exit 2
        fi
        apply=true
        ;;
    *)
        echo "Usage: $0 [--apply]" >&2
        exit 2
        ;;
esac

credentials=$(
    "$root/argus" compose config --environment | python3 -c '
import sys
import base64

values = dict(
    line.rstrip("\n").split("=", 1)
    for line in sys.stdin
    if "=" in line
)

for service in ("API", "CLIO", "PROPHET", "INTELLIGENCE"):
    database = values[f"{service}_DB_NAME"]
    owner = values[f"{service}_DB_USER"]
    password = values[f"{service}_DB_PASSWORD"]

    if not database or not owner or not password:
        raise SystemExit(f"{service}: database, login, and password required")

    password_base64 = base64.b64encode(password.encode("utf-8")).decode("ascii")

    print(f"{database}\t{owner}\t{password_base64}")
'
)

while IFS=$'\t' read -r database owner password_base64; do
    "$root/argus" compose exec -T -e PASSWORD_BASE64="$password_base64" postgres sh -c '
        exec psql \
            --username="$POSTGRES_USER" \
            --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
            --set=ON_ERROR_STOP=1 \
            "$@"
    ' sh --set=database="$database" --set=owner="$owner" --set=apply="$apply" <<'SQL'
        SELECT
            :'database' AS database,
            :'owner' AS expected_owner,
            d.oid IS NOT NULL AS database_exists,
            r.oid IS NOT NULL AS role_exists,
            pg_get_userbyid(d.datdba) AS actual_owner,
            r.rolcanlogin AS login,
            r.rolsuper OR r.rolcreatedb OR r.rolcreaterole
                OR r.rolreplication OR r.rolbypassrls AS admin_privileges
        FROM (SELECT 1) AS seed
        LEFT JOIN pg_database d ON d.datname = :'database'
        LEFT JOIN pg_roles r ON r.rolname = :'owner'
    \gset check_

    \if :check_role_exists
        \if :check_admin_privileges
            DO $$
            BEGIN
                RAISE EXCEPTION 'Service owner has admin privileges';
            END
            $$;
        \endif

        \if :check_login
        \else
            DO $$
            BEGIN
                RAISE EXCEPTION 'Service owner cannot log in';
            END
            $$;
        \endif
    \endif

    \if :check_database_exists
        SELECT
            :'check_actual_owner' = :'check_expected_owner' AS owner_matches
        \gset check_

        \if :check_owner_matches
        \else
            DO $$
            BEGIN
                RAISE EXCEPTION 'Database belongs to another owner';
            END
            $$;
        \endif
    \endif

    \if :check_role_exists
        \echo Role :owner already exists
    \else
        \if :apply
            \getenv argus_password_b64 PASSWORD_BASE64

            SELECT format(
                'CREATE ROLE %I LOGIN PASSWORD %L',
                :'owner',
                convert_from(
                    decode(:'argus_password_b64', 'base64'),
                    'UTF8'
                )
            )
            \gexec

            \echo Created role :owner
        \else
            \echo Will be created role :owner
        \endif
    \endif

    \if :check_database_exists
        \echo Database :database already exists
    \else
        \if :apply
            SELECT format(
                'CREATE DATABASE %I OWNER %I',
                :'database',
                :'owner'
            )
            \gexec

            \echo Created database :database
        \else
            \echo Will be created database :database with owner :owner
        \endif
    \endif
SQL
done <<< "$credentials"