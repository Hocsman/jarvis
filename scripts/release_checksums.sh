#!/bin/bash
# Write SHA256SUMS.txt next to the installers a release publishes.
#
# Usage: release_checksums.sh <artifacts-dir>
#
# Every .zip and .tar.gz under <artifacts-dir> (download-artifact puts each
# one in a sub-directory of its own) is listed by file name only, in the
# format `sha256sum` writes, because the in-app updater looks an installer up
# by the name of the release asset. The file lands in <artifacts-dir> itself.
# It exits non-zero when there is nothing to list: a release with no
# installers has nothing to vouch for.

set -euo pipefail

dir="${1:?usage: release_checksums.sh <artifacts-dir>}"
out="$dir/SHA256SUMS.txt"

if command -v sha256sum >/dev/null 2>&1; then
    digest=(sha256sum)
else
    digest=(shasum -a 256)
fi

: > "$out"
while IFS= read -r asset; do
    (cd "$(dirname "$asset")" && "${digest[@]}" -- "$(basename "$asset")") >> "$out"
done < <(find "$dir" -type f \( -name '*.zip' -o -name '*.tar.gz' \) | sort)

if [ ! -s "$out" ]; then
    echo "❌ No installers under $dir, so there is nothing to checksum" >&2
    rm -f "$out"
    exit 1
fi

echo "🔐 Checksums written to $out"
sed 's/^/   /' "$out"
