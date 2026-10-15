#!/usr/bin/env bash
# Apply an Actions-built release. Compose decides which containers need recreation.
set -euo pipefail
phase_name=preflight
phase_started=$SECONDS
phase() {
    printf '[deploy] %s: %ss\n' "$phase_name" "$((SECONDS - phase_started))"
    phase_name="$1"
    phase_started=$SECONDS
    printf '[deploy] starting %s\n' "$phase_name"
}
trap 'printf "[deploy] %s: %ss; exit=%s\n" "$phase_name" "$((SECONDS - phase_started))" "$?"' EXIT
root="${1:-/var/www}"
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v rsync >/dev/null
command -v gzip >/dev/null
if [[ ${ARGUS_DEPLOY_LOCKED:-0} != 1 ]]; then
    exec 9>"$root/.deployment.lock"
    flock -n 9
fi
base=(docker compose --project-directory "$root" --env-file "$root/.env" --env-file "$root/.env.local")
candidate=("${base[@]}" --env-file "$bundle/images.env" -f "$bundle/docker-compose.yml")
installed=("${base[@]}")
if [[ -f "$root/.release-images.env" ]]; then
    installed+=(--env-file "$root/.release-images.env")
fi
installed+=(-f "$root/docker-compose.yml")
"${candidate[@]}" config --quiet
installed_services="$("${installed[@]}" config --services)"
has_installed_service() {
    [[ $'\n'"$installed_services"$'\n' == *$'\n'"$1"$'\n'* ]]
}

# Compare with the last successful application, never the previous Git tag.
changed() {
    local previous=''
    if [[ -f "$root/.release-fingerprints.tsv" ]]; then
        previous="$(awk -v key="$1" '$1 == key {print $2}' "$root/.release-fingerprints.tsv")"
    fi
    [[ "$previous" != "$(awk -v key="$1" '$1 == key {print $2}' "$bundle/fingerprints.tsv")" ]]
}
migrations=()
for domain in api clio prophet intelligence; do
    if changed "$domain"; then migrations+=("$domain-migrate"); fi
done
retired=()
for service in solar-wind geomagnetic clio-refresh clio-aggregate; do
    if has_installed_service "$service"; then retired+=("$service"); fi
done
phase image-cleanup
# Run under the deployment lock, before pulling candidate images. Keep the
# recent rollback window and every image referenced by a container (even stopped).
if ! docker image prune -a -f --filter 'until=168h'; then
    echo '[deploy] warning: image cleanup failed; continuing deployment' >&2
fi
phase download-images
"${candidate[@]}" pull --policy missing
if ((${#migrations[@]})); then
    "${candidate[@]}" pull --policy missing "${migrations[@]}"
fi
phase check-database-revisions
pending=()
stop=()
for migration in "${migrations[@]}"; do
    domain="${migration%-migrate}"
    status="$("${candidate[@]}" run --rm --no-deps -T "$migration" .venv/bin/python - "$domain" < "$bundle/migration_status.py")"
    case "$status" in
        current) echo "[deploy] $domain: database already at image head" ;;
        pending)
            pending+=("$migration")
            case "$domain" in
                intelligence) stop+=(intelligence intelligence-api) ;;
                api) stop+=(api) ;;
                clio) stop+=(clio clio-worker) ;;
                prophet) stop+=(prophet prophet-api) ;;
            esac
            ;;
        *) echo "Invalid migration status for $domain" >&2; exit 1 ;;
    esac
done
migrations=("${pending[@]}")
printf 'Pending migrations: %s\n' "${migrations[*]:-none}"
if ((${#migrations[@]})); then
    phase database-backup
    python3 "$bundle/backup.py" "$root/backups" -- "${installed[@]}" exec -T postgres sh -c \
        'pg_dumpall --username="$POSTGRES_USER"'
fi
if changed configs; then stop+=(api clio clio-worker prophet prophet-api intelligence-api); fi
stop+=("${retired[@]}")
existing_stop=()
for service in "${stop[@]}"; do
    if has_installed_service "$service"; then existing_stop+=("$service"); fi
done
phase drain-writers
if ((${#existing_stop[@]})); then "${installed[@]}" stop "${existing_stop[@]}"; fi
if ((${#retired[@]})); then "${installed[@]}" rm -f "${retired[@]}"; fi
phase install-configuration
# These directories are repository-owned; data, models and operator env files are separate.
for directory in configs nginx alloy; do
    mkdir -p "$root/$directory"
    rsync -a --delete --inplace "$bundle/$directory/" "$root/$directory/"
done
if [[ -f "$root/.release-compose.json" ]]; then
    install -m 600 "$root/.release-compose.json" "$root/docker-compose.yml"
else
    cp "$bundle/docker-compose.yml" "$root/docker-compose.yml"
fi
cp "$bundle/images.env" "$root/.release-images.env"
mkdir -p "$root/scripts" "$root/bin"
install -m 755 "$bundle/scripts/run" "$root/scripts/run"
install -m 755 "$bundle/scripts/logs" "$root/scripts/logs"
printf 'prod\n' > "$root/.argus-mode"
install -m 755 "$bundle/argus" "$root/argus"
ln -sfn ../argus "$root/bin/argus"
active=("${base[@]}" --env-file "$root/.release-images.env" -f "$root/docker-compose.yml")
# Infrastructure must be healthy before migration; unchanged containers stay running.
phase infrastructure-ready
"${active[@]}" up -d --wait postgres redis
phase migrations
for migration in "${migrations[@]}"; do
    "${active[@]}" run --rm --no-deps "$migration"
done
phase application-ready
"${active[@]}" up -d --wait --wait-timeout 300
phase reload-services
# Bind-mounted configuration changes are not detected by Compose.
for service in nginx alloy; do
    if changed "$service"; then
        "${active[@]}" restart "$service"
    fi
done
# Refresh static upstream DNS after API/frontend containers may have changed.
"${active[@]}" exec -T nginx nginx -t
"${active[@]}" exec -T nginx nginx -s reload
cp "$bundle/fingerprints.tsv" "$root/.release-fingerprints.tsv.next"
mv "$root/.release-fingerprints.tsv.next" "$root/.release-fingerprints.tsv"
cp "$bundle/release.json" "$root/.release.json"
if ((${#migrations[@]})); then
    phase backup-cleanup
    # Only rotate after both a new backup and the deployment have succeeded.
    # Timestamped names sort chronologically, including legacy uncompressed SQL.
    mapfile -t backups < <(find "$root/backups" -maxdepth 1 -type f \
        \( -name 'pre-migration-*.sql' -o -name 'pre-migration-*.sql.gz' \) \
        -printf '%f\n' | LC_ALL=C sort -r)
    for old_backup in "${backups[@]:7}"; do
        rm -- "$root/backups/$old_backup"
    done
    for kept_backup in "${backups[@]:0:7}"; do
        if [[ "$kept_backup" == *.sql ]]; then
            gzip -- "$root/backups/$kept_backup"
        fi
    done
fi
echo 'Release applied successfully'
