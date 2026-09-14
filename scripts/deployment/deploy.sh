#!/usr/bin/env bash
# Apply an Actions-built release. Compose decides which containers need recreation.
set -euo pipefail
root="${1:-/var/www}"
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v rsync >/dev/null
exec 9>"$root/.deployment.lock"
flock -n 9
base=(docker compose --project-directory "$root" --env-file "$root/.env" --env-file "$root/.env.local")
candidate=("${base[@]}" --env-file "$bundle/images.env" -f "$bundle/docker-compose.yml")
installed=("${base[@]}")
if [[ -f "$root/.release-images.env" ]]; then
    installed+=(--env-file "$root/.release-images.env")
fi
installed+=(-f "$root/docker-compose.yml")
"${candidate[@]}" config --quiet

# Compare with the last successful application, never the previous Git tag.
changed() {
    local previous=''
    if [[ -f "$root/.release-fingerprints.tsv" ]]; then
        previous="$(awk -v key="$1" '$1 == key {print $2}' "$root/.release-fingerprints.tsv")"
    fi
    [[ "$previous" != "$(awk -v key="$1" '$1 == key {print $2}' "$bundle/fingerprints.tsv")" ]]
}
migrations=()
stop=()
for domain in api clio prophet; do
    if changed "$domain"; then
        migrations+=("$domain-migrate")
        case "$domain" in
            api) stop+=(api) ;;
            clio) stop+=(clio solar-wind geomagnetic clio-refresh clio-aggregate) ;;
            prophet) stop+=(prophet prophet-api) ;;
        esac
    fi
done
if changed configs; then
    stop=(api clio solar-wind geomagnetic clio-refresh clio-aggregate prophet prophet-api)
fi
printf 'Migrations: %s\n' "${migrations[*]:-none}"
"${candidate[@]}" pull --policy missing
if ((${#migrations[@]})); then
    "${candidate[@]}" pull --policy missing "${migrations[@]}"
fi
if ((${#stop[@]})); then
    "${installed[@]}" stop "${stop[@]}"
fi
if ((${#migrations[@]})); then
    mkdir -p "$root/backups"
    "${installed[@]}" exec -T postgres sh -c \
        'pg_dump --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --format=custom' \
        > "$root/backups/pre-migration-$(date -u +%Y%m%dT%H%M%S).dump"
fi
# These directories are repository-owned; data, models and operator env files are separate.
for directory in configs nginx alloy; do
    mkdir -p "$root/$directory"
    rsync -a --delete --inplace "$bundle/$directory/" "$root/$directory/"
done
cp "$bundle/docker-compose.yml" "$root/docker-compose.yml"
cp "$bundle/images.env" "$root/.release-images.env"
mkdir -p "$root/bin"
install -m 755 "$bundle/argus" "$root/bin/argus"
active=("${base[@]}" --env-file "$root/.release-images.env" -f "$root/docker-compose.yml")
# Infrastructure must be healthy before migration; unchanged containers stay running.
"${active[@]}" up -d --wait postgres redis
for migration in "${migrations[@]}"; do
    "${active[@]}" run --rm --no-deps "$migration"
done
"${active[@]}" up -d --wait --wait-timeout 300
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
echo 'Release applied successfully'
