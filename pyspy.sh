#!/bin/bash

# Configuration
# Prefer venv python if running
# We look for processes running python that match our target
# If setproctitle is used, we might see "3dchess:"

TARGET="$1"

get_pids() {
    # 1. Look for explicitly named processes (setproctitle)
    named_pids=$(pgrep -f "3dchess:")
    
    if [ ! -z "$named_pids" ]; then
        echo "$named_pids"
        return
    fi

    # 2. Look for venv python processes
    # Filter for python processes running from a venv/bin or .venv/bin
    venv_pids=$(ps -eo pid,command | grep -E "venv/bin/python|bin/python" | grep -v grep | awk '{print $1}')
    
    if [ ! -z "$venv_pids" ]; then
        echo "$venv_pids"
        return
    fi
    
    # 3. Fallback: Top CPU python processes
    top_pids=$(ps -eo %cpu,pid,command --no-headers | grep -v grep | grep python | sort -k1 -nr | head -5 | awk '{print $2}')
    echo "$top_pids"
}

select_pid() {
    local pids_list="$1"
    # Convert newline separated PIDs to array
    pids=($pids_list)
    
    if [ ${#pids[@]} -eq 0 ]; then
        echo "No Python processes found."
        exit 1
    fi
    
    if [ ${#pids[@]} -eq 1 ]; then
        # Auto-select if only one
        echo "${pids[0]}"
        return
    fi

    # Interactive selection
    echo "Multiple Python processes found. Please select one:" >&2
    local i=1
    for pid in "${pids[@]}"; do
        # Get command info for the pid
        cmd=$(ps -p "$pid" -o command= | cut -c 1-80)
        cpu=$(ps -p "$pid" -o %cpu=)
        echo "  $i) PID: $pid (CPU: ${cpu}%) - $cmd" >&2
        ((i++))
    done

    # Read user input
    read -p "Enter number (1-${#pids[@]}): " selection
    
    # Validate input (simple check)
    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#pids[@]}" ]; then
         index=$((selection-1))
         echo "${pids[$index]}"
    else
         echo "Invalid selection." >&2
         exit 1
    fi
}

echo "Scanning for processes..."

if [ "$TARGET" == "server" ]; then
    # specifically look for server tag
    pids=$(pgrep -f "3dchess: server")
elif [ "$TARGET" == "worker" ]; then
    # specifically look for worker tag
    pids=$(pgrep -f "3dchess: worker")
else
    # General scan
    pids=$(get_pids)
fi

target_pid=$(select_pid "$pids")

echo ""
echo "Targeting PID: $target_pid"
cmd_name=$(ps -p "$target_pid" -o command=)
echo "Command: $cmd_name"
echo "Starting py-spy..."
echo ""

sudo py-spy top --pid "$target_pid" --rate 10
