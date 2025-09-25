#!/bin/bash

# Check if a directory is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY="$1"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY does not exist"
    exit 1
fi

# Find all Python files in the directory
find "$DIRECTORY" -name "*.py" -type f | while read -r file; do
    echo "Processing $file..."

    # Create a temporary file
    temp_file=$(mktemp)

    # Process the file
    awk '
    BEGIN {
        in_function = 0
        processed = 0
    }
    {
        # Check if this line contains the function signature we want to replace
        if (match($0, /\(state: GameState, x: int, y: int, z: int\) -> List\[Move\]:/)) {
            # Extract the indentation
            indent = substr($0, 1, RSTART - 1)

            # Print the new function signature with the same indentation
            print indent "(board, color, *coord, cache=None) -> List[Move]:"

            # Add the import statement after the function signature
            print indent "    from game3d.game.gamestate import GameState"
            print indent "    state = GameState(board, color, cache=cache)"

            in_function = 2  # We need to skip the next line too if it exists
            processed = 1
        } else {
            if (in_function > 0) {
                in_function--
            } else {
                print $0
            }
        }
    }
    END {
        if (processed) {
            # If we made changes, the exit status should indicate success for this file
            exit 0
        }
    }' "$file" > "$temp_file"

    # Move the temporary file to the original file
    mv "$temp_file" "$file"
done

echo "Processing complete!"
