#!/bin/bash
# Fast SAPIEN headless-render pre-check (~10 min): install just miniconda +
# pytorch + sapien, then try an offscreen render. Decides whether this host's
# driver can render before we commit to the full 40-min setup.
exec > /root/precheck.log 2>&1
set -x
apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 wget vulkan-tools >/dev/null 2>&1
if [ ! -d /opt/miniconda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p /opt/miniconda && rm /tmp/mc.sh
fi
export PATH=/opt/miniconda/bin:$PATH
pip install -q torch==2.5.1+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip install -q "numpy<2" sapien==2.2.2 2>&1 | tail -1
export VK_ICD_FILENAMES=$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
echo "VK_ICD=$VK_ICD_FILENAMES ; driver:"; nvidia-smi --query-gpu=driver_version --format=csv,noheader
cat > /root/saptest.py <<'PY'
import sapien.core as sapien
e=sapien.Engine(); r=sapien.SapienRenderer(); e.set_renderer(r)
sc=e.create_scene(); sc.add_ground(0)
cam=sc.add_camera("c",64,64,1,0.1,10)
sc.step(); sc.update_render(); cam.take_picture()
print("SAPIEN_RENDER_OK")
PY
python /root/saptest.py 2>&1 | tail -6
if grep -q SAPIEN_RENDER_OK /root/precheck.log; then touch /root/RENDER_OK; echo PRECHECK_PASS; else touch /root/RENDER_FAIL; echo PRECHECK_FAIL; fi
