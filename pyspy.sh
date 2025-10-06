#!/bin/zsh

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

echo "Monitoring top Python process (PID: $top_pid) with highest CPU usage..."
sudo py-spy top --pid "$top_pid" --rate 10
