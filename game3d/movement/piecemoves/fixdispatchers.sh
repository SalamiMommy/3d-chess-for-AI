#!/bin/bash

# replace_moves_state.sh
# Replace moves(state, calls with moves(state.board, state.cache, state.color,

set -e  # Exit immediately if a command exits with a non-zero status

# Default directory is current directory, but you can pass one as argument
TARGET_DIR="${1:-.}"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    echo "Usage: $0 [directory_path]"
    exit 1
fi

echo "Searching and replacing in directory: $TARGET_DIR"
echo "----------------------------------------"

# Create backup of files that will be modified
echo "Creating backups (with .bak extension)..."
find "$TARGET_DIR" -type f -name "*.py" -exec grep -l "moves(state," {} \; | while read -r file; do
    if [ -f "$file" ]; then
        cp "$file" "$file.bak"
        echo "Backup created: $file.bak"
    fi
done

# Perform the replacement
echo "Performing replacement: moves(state, → moves(state.board, state.cache, state.color,"

# Replace moves(state, calls
find "$TARGET_DIR" -type f -name "*.py" -exec sed -i 's/moves(state,/moves(state.board, state.cache, state.color,/g' {} +

echo "----------------------------------------"
echo "Replacement completed successfully!"

# Show which files were modified
echo "Files that were modified:"
find "$TARGET_DIR" -type f -name "*.py" -exec grep -l "moves(state.board, state.cache, state.color," {} \;

echo ""
echo "Note: Original files have been backed up with .bak extension"
echo "To restore: rename .bak files back to original names"
echo "To clean up backups: find . -name \"*.bak\" -delete"
