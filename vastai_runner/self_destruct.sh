#!/bin/bash
# Add this to the end of remote sweep launch scripts.
# After the main job finishes, it syncs results back and destroys the instance.
#
# Usage (on the remote machine):
#   nohup bash self_destruct.sh <LOCAL_USER> <LOCAL_HOST> <LOCAL_PORT> <INSTANCE_ID> &
#
# Or source the function:
#   source self_destruct.sh
#   wait_and_destroy <pids_to_wait_for...>

wait_and_destroy() {
    local pids=("$@")

    echo "[self-destruct] Waiting for PIDs: ${pids[*]}"
    for pid in "${pids[@]}"; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 60
        done
        echo "[self-destruct] PID $pid finished"
    done

    echo "[self-destruct] All jobs done. Uploading results..."

    # Try to sync results back (best-effort)
    if [ -d /root/project/results ]; then
        tar czf /tmp/results_$(date +%Y%m%d_%H%M).tar.gz -C /root/project results/
        echo "[self-destruct] Results archived at /tmp/results_*.tar.gz"
    fi

    echo "[self-destruct] Jobs complete. Instance can be destroyed."
    # The local watchdog will catch this as idle and destroy it,
    # or uncomment below if you have vastai CLI on the remote:
    # pip install vastai && vastai destroy instance $(cat /etc/hostname | head -1) || true
}

# If run directly with PIDs as args
if [ $# -gt 0 ]; then
    wait_and_destroy "$@"
fi
