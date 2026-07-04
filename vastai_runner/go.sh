#!/bin/bash
# Combined launcher: full setup, then run_job.sh (sdpa fix + server + frozen LOO).
# Render already proven on this host (driver 535). Writes /root/FAILED on setup error.
exec > /root/go.log 2>&1
set -x
bash /root/setup_nvvulkan.sh || { echo "SETUP_FAILED"; touch /root/FAILED; exit 1; }
echo "SETUP_DONE -> launching job"
bash /root/run_job.sh
