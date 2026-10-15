#!/usr/bin/env bash
set -euo pipefail
root="${1:-/var/www}"
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v python3 >/dev/null
exec 8>"$root/.deployment.lock"
flock -n 8
python3 "$bundle/rolling.py" "$root" "$bundle" --preflight
if [[ ! -f "$root/.rolling-state.json" ]]; then
    echo '[deploy] Initial transition: applying the legacy release before installing HTTP routers.'
    ARGUS_DEPLOY_LOCKED=1 bash "$bundle/deploy-legacy.sh" "$root"
fi
python3 "$bundle/rolling.py" "$root" "$bundle"
