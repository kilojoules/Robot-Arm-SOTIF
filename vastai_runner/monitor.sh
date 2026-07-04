#!/bin/bash
# Local monitor (v4) for the frozen-ResNet LOO run.
#
# Cleanup authority lives here (account key works locally; the instance-scoped
# key does NOT authorize destroy, so box self-destruct is unavailable).
# Two cleanup paths, neither acts on transient API status strings:
#   1) explicit /root/DONE|FAILED sentinel via SSH -> pull results + destroy
#   2) backstop: gpu_util==0 for >=IDLE_CHECKS consecutive runs AND uptime>GRACE
#      -> best-effort pull + destroy (catches done/crashed/hung box)
# Runs from cron every 20 min; survives laptop sleep (resumes on wake).
#
#   */20 * * * * cd <dir> && bash monitor.sh >> /tmp/vast_monitor.log 2>&1

set -uo pipefail
cd "$(dirname "$0")"

VASTAI=".pixi/envs/default/bin/vastai"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
RESULTS_LOCAL="/Users/julianquick/portfolio_copy/TRI_pet_project/results/loo_analysis_frozen_vast"
GRACE_MIN=240        # never backstop-kill in first 4h (setup + warmup)
IDLE_CHECKS=6        # ~2h of sustained gpu=0 after grace before backstop kill
TS="[monitor $(date '+%Y-%m-%d %H:%M')]"

iid=$(cat .active_instance 2>/dev/null || echo "")
if [ -z "$iid" ]; then echo "$TS no .active_instance; idle"; exit 0; fi

inst=$("$VASTAI" show instances --raw 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except: d=[]
d=d if isinstance(d,list) else [d]
m=[i for i in d if str(i.get('id'))=='$iid']
print(json.dumps(m[0]) if m else '')
" 2>/dev/null)

if [ -z "$inst" ]; then
    echo "$TS instance $iid gone. Clearing state."; rm -f .active_instance .idle_count; exit 0
fi

read uptime_min gpu host port <<<$(echo "$inst" | python3 -c "
import json,sys,time
i=json.load(sys.stdin); now=time.time(); sd=i.get('start_date') or now
g=i.get('gpu_util'); g=0 if g is None else int(g)
print(f'{(now-sd)/60:.0f}', g, i.get('ssh_host','-'), i.get('ssh_port','-'))
")
echo "$TS id=$iid uptime=${uptime_min}min gpu=${gpu}% ssh=$host:$port"

pull() { mkdir -p "$RESULTS_LOCAL"
    rsync -az -e "ssh -i $SSH_KEY -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=25" \
        "root@$host:/root/project/results/" "$RESULTS_LOCAL/" 2>&1 | tail -2 || true
    scp -i "$SSH_KEY" -P "$port" -o StrictHostKeyChecking=no "root@$host:/root/loo_results.tar.gz" "$RESULTS_LOCAL/" 2>&1 | tail -1 || true
    for L in setup.log server.log run_frozen.log; do
        scp -i "$SSH_KEY" -P "$port" -o StrictHostKeyChecking=no "root@$host:/root/$L" "$RESULTS_LOCAL/$L" 2>/dev/null || true
    done; }
destroy() { "$VASTAI" destroy instance "$iid" 2>&1 || true; rm -f .active_instance .idle_count; }

# 1) sentinel via SSH (primary)
if [ "$host" != "-" ]; then
    state=$(ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes \
            root@"$host" 'ls /root/DONE /root/FAILED 2>/dev/null' 2>/dev/null || echo "")
    if echo "$state" | grep -qE "/root/(DONE|FAILED)"; then
        echo "$TS sentinel found -> pull results + destroy"; pull; destroy; echo "$TS cleaned up."; exit 0
    fi
fi

# 2) conservative sustained-idle backstop (no action on status strings)
if [ "$uptime_min" -gt "$GRACE_MIN" ] && [ "$gpu" = "0" ]; then
    c=$(( $(cat .idle_count 2>/dev/null || echo 0) + 1 )); echo "$c" > .idle_count
    echo "$TS sustained-idle $c/$IDLE_CHECKS (gpu=0, past grace)"
    if [ "$c" -ge "$IDLE_CHECKS" ]; then
        echo "$TS idle backstop tripped -> pull + destroy"; pull; destroy; exit 0
    fi
else
    rm -f .idle_count
fi
echo "$TS OK (running)"
