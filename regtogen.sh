#!/usr/bin/env bash
# fix_registry.sh  –  replace every occurrence of
#   game3d.movement.registry
# with
#   game3d.movement.generator
# in every *.py file under the given directory tree.

set -euo pipefail

ROOT_DIR="${1:-.}"
FROM='game3d\.movement\.registry'
TO='game3d.movement.generator'

# Portable -i: create a backup extension that we delete immediately
find "$ROOT_DIR" -type f -iname '*.py' -print0 |
while IFS= read -r -d '' file; do
    sed -i.bak "s/${FROM}/${TO}/g" "$file"
    rm -f "${file}.bak"
done

echo "Done – replaced '${FROM//\\/}' with '${TO}' in all Python files under ${ROOT_DIR}"
