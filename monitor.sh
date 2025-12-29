#!/bin/bash

# ==============================================================================
# MLX MEMORY MONITOR (Auto-Start/Stop)
# Usage: ./monitor_auto.sh your_script.py
# ==============================================================================

SCRIPT_NAME=$1

if [ -z "$SCRIPT_NAME" ]; then
    echo "Usage: $0 <script_name.py>"
    exit 1
fi

# Variables to track state
HAS_STARTED=false
PEAK_KB=0
TOTAL_ACC_KB=0
SAMPLES=0
START_TIME=0

# Helper to clear screen
clear_screen() {
    printf "\033c"
}

clear_screen
echo "--- MLX Monitor Ready ---"
echo "Waiting for process: 'python ... $SCRIPT_NAME' to start..."

# Main Loop
while true; do
    # 1. Find PIDs (Launcher + Workers)
    #    We use 'pgrep -f' to find the script, but verify it matches 'python'
    #    tr/sed converts the list to "123,456,789" format for 'ps'
    PIDS=$(pgrep -f "python.*$SCRIPT_NAME" | tr '\n' ',' | sed 's/,$//')

    # --------------------------------------------------------------------------
    # STATE: WAITING TO START
    # --------------------------------------------------------------------------
    if [ -z "$PIDS" ]; then
        if [ "$HAS_STARTED" = true ]; then
            # We were running, but now PIDs are gone -> Job Finished.
            break
        else
            # Still waiting for the user to run the command
            sleep 1
            continue
        fi
    fi

    # --------------------------------------------------------------------------
    # STATE: RUNNING (First detection)
    # --------------------------------------------------------------------------
    if [ "$HAS_STARTED" = false ]; then
        HAS_STARTED=true
        START_TIME=$(date +%s)
        clear_screen
        echo "✅ Detected MLX Job! Starting monitor..."
        sleep 1
    fi

    # --------------------------------------------------------------------------
    # DATA COLLECTION
    # --------------------------------------------------------------------------
    # Sum current RSS (Resident Set Size) in KB for all detected PIDs
    CURRENT_KB=$(ps -p "$PIDS" -o rss | awk 'NR>1 {sum+=$1} END {print sum}')
    
    # Safety check if query returned empty (process died mid-loop)
    if [ -z "$CURRENT_KB" ]; then CURRENT_KB=0; fi

    # Update Statistics (Bash handles integers well, so we keep KB)
    if [ "$CURRENT_KB" -gt "$PEAK_KB" ]; then
        PEAK_KB=$CURRENT_KB
    fi

    TOTAL_ACC_KB=$((TOTAL_ACC_KB + CURRENT_KB))
    SAMPLES=$((SAMPLES + 1))

    # Calculate Runtime
    NOW=$(date +%s)
    RUNTIME=$((NOW - START_TIME))

    # --------------------------------------------------------------------------
    # DISPLAY DASHBOARD
    # --------------------------------------------------------------------------
    clear_screen
    echo "=== MLX LIVE MONITOR: $SCRIPT_NAME ==="
    printf "Runtime: %ds  |  Samples: %d\n" "$RUNTIME" "$SAMPLES"
    echo "----------------------------------------------------------------"
    printf "%-8s | %-12s | %-12s | %s\n" "PID" "ROLE" "MEMORY" "COMMAND"
    echo "----------------------------------------------------------------"

    # Pass stats to awk for pretty printing
    ps -p "$PIDS" -o pid,rss,command | awk -v script="$SCRIPT_NAME" \
                                          -v peak_kb="$PEAK_KB" \
                                          -v acc_kb="$TOTAL_ACC_KB" \
                                          -v samples="$SAMPLES" '
    BEGIN {
        worker_mem = 0
        total_mem = 0
    }
    NR>1 {
        # Process Row Data
        pid = $1
        mem_mb = $2 / 1024
        total_mem += mem_mb

        # Identify Orchestrator vs Worker
        if ($0 ~ /mlx.launch/) {
            role = "ORCHESTRATOR"
            color = "\033[36m" # Cyan
        } else {
            role = "WORKER"
            color = "\033[32m" # Green
            worker_mem += mem_mb
        }
        reset = "\033[0m"
        
        # Format Command
        cmd = "python " script "..."
        
        printf "%s%-8s | %-12s | %9.2f MB | %s%s\n", color, pid, role, mem_mb, cmd, reset
    }
    END {
        print "----------------------------------------------------------------"
        
        # Calculate Stats in GB
        cur_gb = total_mem / 1024
        peak_gb = peak_kb / 1024 / 1024
        
        # Calculate Average
        if (samples > 0) {
            avg_kb = acc_kb / samples
            avg_gb = avg_kb / 1024 / 1024
        } else {
            avg_gb = 0
        }

        printf "Current Total:       %9.2f GB\n", cur_gb
        printf "Average Total:       %9.2f GB\n", avg_gb
        printf "\033[1;31mPeak Total:          %9.2f GB\033[0m\n", peak_gb
    }'

    sleep 1
done

# --------------------------------------------------------------------------
# STATE: FINISHED (Final Report)
# --------------------------------------------------------------------------
clear_screen
echo "=== 🏁 JOB FINISHED: $SCRIPT_NAME ==="
echo "Monitoring stopped because processes exited."
echo "----------------------------------------"

# Re-calculate final stats for summary using awk (Bash float math is hard)
awk -v peak="$PEAK_KB" -v acc="$TOTAL_ACC_KB" -v samp="$SAMPLES" 'BEGIN {
    peak_gb = peak / 1024 / 1024
    if (samp > 0) avg_gb = (acc / samp) / 1024 / 1024
    else avg_gb = 0
    
    printf "⏱  Total Runtime:  %d seconds\n", samp
    printf "📈 Average Memory: %.2f GB\n", avg_gb
    printf "🚀 Peak Memory:    %.2f GB\n", peak_gb
}'
echo "----------------------------------------"