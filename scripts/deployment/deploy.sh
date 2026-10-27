#!/usr/bin/env bash
set -euo pipefail
root="${1:-/var/www}"
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$root/.rolling-state.json" ]]; then
    echo '[deploy] Missing .rolling-state.json; restore deployment state before applying a release.' >&2
    exit 1
fi
command -v python3 >/dev/null
command -v crontab >/dev/null
exec 8>"$root/.deployment.lock"
flock -n 8
python3 "$bundle/rolling.py" "$root" "$bundle"
