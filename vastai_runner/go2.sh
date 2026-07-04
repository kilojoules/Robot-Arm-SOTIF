#!/bin/bash
# One-shot: corrected slim setup -> render gate -> frozen+finetune LOO.
exec > /root/go2.log 2>&1
set -x
bash /root/slim_setup.sh
if [ -f /root/SLIM_RENDER_OK ]; then
    echo "RENDER_OK -> launching both LOOs"
    bash /root/run_both.sh
else
    echo "RENDER_FAILED_AT_SETUP -> abort"; touch /root/FAILED
fi
