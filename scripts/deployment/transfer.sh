#!/usr/bin/env bash
# One-time split of the existing shared database, followed by release application.
set -euo pipefail
root="${1:?project root required}"
shift
state="$root/backups/database-transfer"
release=''
source=''
while (($#)); do
    case "$1" in
        --release|--source)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            if [[ "$1" == --release ]]; then release="$2"; else source="$2"; fi
            shift 2 ;;
        -h|--help)
            echo 'Usage: argus db transfer [--release /absolute/release-bundle] [--source old_database]'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ ! "$root/.deployment.lock" -ef /proc/self/fd/9 ]]; then exec 9>"$root/.deployment.lock"; fi
flock -xn 9
if [[ -z "$release" ]]; then
    if [[ -f "$state/bundle" ]]; then read -r release < "$state/bundle"
    elif [[ -f "$root/.prepared-release" ]]; then read -r release < "$root/.prepared-release"
    else echo 'Prepare a release with prepare_only=true first, or supply --release.' >&2; exit 1
    fi
fi
release="$(cd -- "$release" && pwd)"
for file in deploy.sh images.env docker-compose.yml release.json fingerprints.tsv; do
    [[ -f "$release/$file" ]] || { echo "Incomplete release: missing $file" >&2; exit 1; }
done
umask 077
mkdir -p "$state"
chmod 700 "$state"
if [[ -f "$state/bundle" ]]; then
    read -r previous < "$state/bundle"
    [[ "$release" == "$previous" ]] || { echo "Resume the original release: $previous" >&2; exit 1; }
    for file in images.env docker-compose.yml fingerprints.tsv; do
        cmp -s "$release/$file" "$state/release-$file" || { echo 'Prepared release changed during transfer' >&2; exit 1; }
    done
else
    for file in images.env docker-compose.yml fingerprints.tsv; do cp "$release/$file" "$state/release-$file"; done
    printf '%s\n' "$release" > "$state/bundle.next"
    mv "$state/bundle.next" "$state/bundle"
fi
if [[ -z "$source" && -f "$state/source" ]]; then read -r source < "$state/source"; fi
base=(docker compose --project-directory "$root" --env-file "$root/.env" --env-file "$root/.env.local")
candidate=("${base[@]}" --env-file "$release/images.env" -f "$release/docker-compose.yml")
installed=("${base[@]}" --env-file "$root/.release-images.env" -f "$root/docker-compose.yml")
system_identifier="$("${installed[@]}" exec -T postgres sh -c 'psql --username="$POSTGRES_USER" --dbname=postgres --no-psqlrc -Atc "SELECT system_identifier FROM pg_control_system()"')"
[[ "$system_identifier" =~ ^[0-9]+$ ]] || { echo 'Cannot identify the installed PostgreSQL server' >&2; exit 1; }
helper() {
    local args=()
    [[ -z "$source" ]] || args+=(--source "$source")
    "${candidate[@]}" run --rm --no-deps -T --volume "$state:/transfer" db-provision \
        .venv/bin/python /var/www/scripts/transfer-databases.py "$@" "${args[@]}" --system-identifier "$system_identifier"
}
trap 'status=$?; if ((status)); then echo "Transfer stopped. Checkpoints and backups: $state. Writers may remain stopped; rerun the same db transfer command after fixing the error." >&2; fi' EXIT
"${candidate[@]}" config --quiet
"${candidate[@]}" pull --policy missing
"${candidate[@]}" pull --policy missing db-provision api-migrate clio-migrate prophet-migrate intelligence-migrate
if [[ -f "$state/verified" ]]; then
    # A failed deployment may already have started writers in the new databases.
    # Never recopy old data or compare it against databases that now accept writes.
    helper identity
else
    helper preflight > "$state/plan.tsv.next"
    mv "$state/plan.tsv.next" "$state/plan.tsv"
    IFS=$'\t' read -r header source < "$state/plan.tsv"
    [[ "$header" == source && -n "$source" ]] || { echo 'Invalid transfer plan' >&2; exit 1; }
    printf '%s\n' "$source" > "$state/source"
    if [[ ! -d "$state/original" ]]; then
        mkdir -p "$state/original.next"
        for file in .env .env.local docker-compose.yml .release-images.env .release-fingerprints.tsv .release.json; do
            [[ ! -f "$root/$file" ]] || cp "$root/$file" "$state/original.next/$file"
        done
        mv "$state/original.next" "$state/original"
    fi
    echo '[transfer] stopping application services'
    services=()
    while read -r service; do
        case "$service" in
            api|clio|solar-wind|geomagnetic|clio-refresh|clio-aggregate|prophet|prophet-api|intelligence) services+=("$service") ;;
        esac
    done < <("${installed[@]}" config --services)
    ((${#services[@]})) || { echo 'No installed application services found' >&2; exit 1; }
    "${installed[@]}" stop "${services[@]}"
    helper freeze
    if [[ ! -f "$state/cluster.sql" ]]; then
        echo '[transfer] backing up PostgreSQL'
        "${installed[@]}" exec -T postgres sh -c 'pg_dumpall --username="$POSTGRES_USER"' > "$state/cluster.sql.next"
        mv "$state/cluster.sql.next" "$state/cluster.sql"
    fi
    helper provision
    while IFS=$'\t' read -r domain database owner ready; do
        [[ "$domain" != source ]] || continue
        echo "[transfer] snapshot and verification: $domain"
        helper snapshot --domain "$domain"
        if [[ "$ready" != 1 ]]; then
            "${installed[@]}" exec -T postgres sh -c \
                'pg_dump --username="$POSTGRES_USER" --dbname="$1" --schema="$2" --format=custom --no-owner --no-privileges' \
                sh "$source" "$domain" > "$state/$domain.dump.next"
            mv "$state/$domain.dump.next" "$state/$domain.dump"
            "${installed[@]}" exec -T postgres sh -c \
                'pg_restore --username="$POSTGRES_USER" --dbname="$1" --role="$2" --no-owner --no-privileges --exit-on-error --single-transaction' \
                sh "$database" "$owner" < "$state/$domain.dump"
        fi
        helper verify --domain "$domain"
    done < "$state/plan.tsv"
    touch "$state/verified"
fi
if [[ -f "$state/completed" ]]; then echo 'Database transfer already completed.'; exit 0; fi
echo '[transfer] applying prepared release'
bash "$release/deploy.sh" "$root"
touch "$state/completed"
echo "Database transfer completed. Source retained read-only; backups: $state"
