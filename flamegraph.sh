#!/bin/zsh

# --- Configuration ---
# Set the output filename
OUTPUT_FILE="flamegraph_output_full_run.svg"
# ---------------------

# Get the PID of the Python process with highest CPU%
# -o specifies custom output format: %cpu and pid
# --no-headers removes column headers
# sort -k1 -nr sorts by first column (CPU%) numerically in reverse order
# head -1 gets the top process
# awk extracts just the PID (2nd field)
top_pid=$(ps -eo %cpu,pid,command --no-headers | grep -v grep | grep python | sort -k1 -nr | head -1 | awk '{print $2}')

if [ -z "$top_pid" ]; then
    echo "No Python processes found."
    exit 1
fi

echo "Starting **py-spy record** on top Python process (PID: $top_pid)."
echo "Profiling will continue until the Python process exits OR you press **Ctrl+C**."
echo "Output will be saved to: $OUTPUT_FILE"

# The updated command:
# 1. Uses 'record' instead of 'top'
# 2. Uses -o to specify the SVG output file
# 3. The **--duration flag is removed** so it runs indefinitely.
sudo py-spy record \
    --pid "$top_pid" \
    --rate 10 \
    --output "$OUTPUT_FILE"

echo "Profiling finished. Flame Graph generation complete!"
echo "Open $OUTPUT_FILE in your browser to view the profile of the entire execution."
