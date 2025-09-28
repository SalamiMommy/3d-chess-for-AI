#!/usr/bin/env bash
# fix_imports.sh  –  fix every Python file under the current directory
# usage:  ./fix_imports.sh

set -euo pipefail

# the exact strings we search / replace
OLD='from game3d.movement.generator import register'
NEW='from game3d.movement.registry import register'

# find every *.py file and apply the substitution in-place
find . -type f -name '*.py' -print0 |
while IFS= read -r -d '' file; do
    # only touch files that actually contain the old line
    if grep -qF "$OLD" "$file"; then
        sed -i "s|$OLD|$NEW|g" "$file"
        echo "fixed  $file"
    fi
done

echo "All done."
