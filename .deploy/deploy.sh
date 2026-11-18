#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(dirname "$0")}"

for tool in python3 crontab flock docker rsync gzip; do
    if ! command -v "$tool" >/dev/null; then
        echo "[deploy] Required command is unavailable: $tool" >&2
        exit 1
    fi
done

lock_timeout="${DEPLOY_LOCK_TIMEOUT:-600}"
if [[ ! "$lock_timeout" =~ ^[0-9]+$ ]]; then
    echo '[deploy] DEPLOY_LOCK_TIMEOUT must be a nonnegative number of seconds.' >&2
    exit 2
fi

exec 8>"$root/.deployment.lock"
echo "[deploy] Waiting for deployment/maintenance lock (up to ${lock_timeout}s)..."

if flock -w "$lock_timeout" -E 75 8; then
    echo '[deploy] Deployment/maintenance lock acquired.'
else
    status=$?
    echo "[deploy] Could not acquire deployment/maintenance lock (status $status); another deployment or maintenance job may still be running." >&2
    exit "$status"
fi

release_id=$(< "$root/.release")
bundle="$root/incoming/$release_id"
if [[ ! -f "$bundle/scripts/deployment/rollout.py" ]]; then
    echo "[deploy] Release bundle is unavailable: $bundle" >&2
    exit 1
fi

python3 "$bundle/scripts/deployment/rollout.py" "$root" "$bundle"
