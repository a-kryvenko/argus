#!/usr/bin/env bash
# Prepare-only releases update operator commands without applying application changes.
set -euo pipefail
root="${1:-/var/www}"
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"$root/.deployment.lock"
flock -xn 9
mkdir -p "$root/bin" "$root/.deployment-tools"
install -m 755 "$bundle/transfer.sh" "$root/.deployment-tools/transfer.sh.next"
mv "$root/.deployment-tools/transfer.sh.next" "$root/.deployment-tools/transfer.sh"
install -m 755 "$bundle/argus" "$root/bin/argus.next"
mv "$root/bin/argus.next" "$root/bin/argus"
printf '%s\n' "$bundle" > "$root/.prepared-release.next"
mv "$root/.prepared-release.next" "$root/.prepared-release"
echo 'Release prepared. Run /var/www/bin/argus db transfer to transfer the existing database and apply it.'
