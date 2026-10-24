#!/usr/bin/env bash
# Builds this app and vendors the bundle into a checkout of the `sites` repo,
# which only publishes it (GitHub Pages). This repo stays the source of truth.
#
#   scripts/publish-sites.sh ../mono    # then review, commit and push in sites
set -euo pipefail

sites=${1:?usage: publish-sites.sh <path-to-sites-checkout>}
sites=$(cd "$sites" && pwd)
dest="$sites/shader-editor"
root=$(cd "$(dirname "$0")/.." && pwd)

cd "$root"
npm run build
rm -rf "$dest"
mkdir -p "$dest"
(cd dist && find . -type f ! -name '*.map' -exec cp --parents {} "$dest" \;)
echo "shader-editor @ $(git -C "$root" rev-parse --short HEAD) -> $dest"
