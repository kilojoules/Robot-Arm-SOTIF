#!/bin/bash
# Vast.ai idle instance watchdog
# Destroys instances that have been idle (no GPU utilization) for too long,
# or that are in "exited" state (still billing but not running).
#
# Usage:
#   # One-shot check:
#   bash watchdog.sh
#
#   # Cron every 30 min (add with: crontab -e)
#   */30 * * * * cd /Users/julianquick/TRI_pet_project/vastai_runner && bash watchdog.sh >> /tmp/vastai_watchdog.log 2>&1

set -euo pipefail
cd "$(dirname "$0")"

VASTAI=".pixi/envs/default/bin/vastai"
MAX_IDLE_HOURS=2  # destroy if GPU util=0% for this long
LOG_PREFIX="[vastai-watchdog $(date '+%Y-%m-%d %H:%M')]"

# Get all instances as JSON
instances=$($VASTAI show instances --raw 2>/dev/null || echo "[]")

if [ "$instances" = "[]" ]; then
    echo "$LOG_PREFIX No instances running."
    exit 0
fi

echo "$instances" | python3 -c "
import json, sys

data = json.load(sys.stdin)
if not isinstance(data, list):
    data = [data]

for inst in data:
    iid = inst.get('id', '?')
    status = inst.get('actual_status', inst.get('status_msg', 'unknown'))
    gpu_util = inst.get('gpu_util', 0) or 0
    uptime_mins = inst.get('duration', 0) or 0  # uptime in minutes
    cost_hr = inst.get('dph_total', 0) or 0
    label = inst.get('label', '')

    # Skip instances labeled 'keep' (manual override)
    if label and 'keep' in label.lower():
        print(f'$LOG_PREFIX Instance {iid}: labeled keep, skipping')
        continue

    # Destroy exited instances (still billing but not usable)
    if 'exited' in status.lower() or 'offline' in status.lower():
        print(f'$LOG_PREFIX Instance {iid}: status={status}, DESTROYING (not running but billing)')
        print(f'  ACTION: destroy {iid}')
        continue

    # Destroy instances with 0% GPU util running longer than MAX_IDLE_HOURS
    if gpu_util == 0 and uptime_mins > ${MAX_IDLE_HOURS} * 60:
        print(f'$LOG_PREFIX Instance {iid}: gpu_util=0%, uptime={uptime_mins:.0f}min, DESTROYING (idle too long)')
        print(f'  ACTION: destroy {iid}')
        continue

    print(f'$LOG_PREFIX Instance {iid}: status={status}, gpu={gpu_util}%, uptime={uptime_mins:.0f}min, \${cost_hr:.2f}/hr - OK')
" | while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == *"ACTION: destroy"* ]]; then
        instance_id=$(echo "$line" | grep -o '[0-9]*')
        $VASTAI destroy instance "$instance_id" 2>&1 || true
    fi
done
