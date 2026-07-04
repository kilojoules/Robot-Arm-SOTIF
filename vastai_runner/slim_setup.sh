#!/bin/bash
# SLIM setup for the InternVLA-M1 LOO pipeline — deliberately OMITS JAX,
# TensorFlow, and Octo (only needed for the Octo policy; their nvidia-cuda
# pip wheels are the prime suspect for clobbering svulkan2 rendering).
# Ends with a SimplerEnv render test -> /root/SLIM_RENDER_OK | SLIM_RENDER_FAIL.
exec > /root/slim_setup.log 2>&1
set -x
apt-get update -qq
apt-get install -y -qq git wget libgl1-mesa-glx libglib2.0-0 ffmpeg cmake build-essential tmux vulkan-tools >/dev/null 2>&1

if [ ! -d /opt/miniconda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p /opt/miniconda && rm /tmp/mc.sh
fi
export PATH=/opt/miniconda/bin:$PATH

pip install torch==2.5.1+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip install "numpy<2" 2>&1 | tail -1
pip install sapien==2.2.2 gymnasium ruckig 2>&1 | tail -1

cd /root
[ ! -d SimplerEnv ] && git clone https://github.com/simpler-env/SimplerEnv
pip install -e SimplerEnv/ --no-deps 2>&1 | tail -1
[ ! -d ManiSkill2_real2sim ] && git clone https://github.com/simpler-env/ManiSkill2_real2sim
pip install -e ManiSkill2_real2sim/ 2>&1 | tail -1
[ ! -d InternVLA-M1 ] && git clone https://github.com/InternRobotics/InternVLA-M1

# Runtime deps for InternVLA + SimplerEnv + the safety pipeline (NO jax/tf/octo)
pip install "transformers==4.52.3" dacite pyyaml cma matplotlib imageio einops websockets \
    qwen-vl-utils omegaconf accelerate timm transforms3d "setuptools<71" pytest \
    "opencv-python-headless<4.11" huggingface-hub scipy scikit-learn mediapy \
    sentencepiece protobuf msgpack msgpack-numpy rich draccus jsonlines 2>&1 | tail -1
# peft/diffusers WITHOUT deps: their python code is available for inference but
# they do NOT pull nvidia-cuda-* wheels (those shadow system CUDA and segfault svulkan2)
pip install --no-deps peft diffusers 2>&1 | tail -1

# sdpa fix (no flash-attn)
sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' \
    /root/InternVLA-M1/InternVLA/model/modules/vlm/QWen2_5.py

# checkpoint
if [ ! -f /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt ]; then
    python -c "
from huggingface_hub import hf_hub_download; import os
os.makedirs('/root/internvla_m1_ckpt/checkpoints',exist_ok=True)
for f in ['checkpoints/steps_50000_pytorch_model.pt','config.yaml','dataset_statistics.json']:
    hf_hub_download(repo_id='InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge',filename=f,local_dir='/root/internvla_m1_ckpt')
print('ckpt ok')"
    sed -i 's/attn_implementation: flash_attention_2/attn_implementation: sdpa/' /root/internvla_m1_ckpt/config.yaml 2>/dev/null || true
fi

cd /root
[ ! -d camera_occlusion ] && git clone -b feature/envelope-predictor https://github.com/kilojoules/camera_occlusion
pip install -e camera_occlusion/ --no-deps 2>&1 | tail -1
[ ! -d project ] && git clone https://github.com/kilojoules/Robot-Arm-SOTIF project

ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true
pip install "numpy<2" 2>&1 | tail -1   # reassert after any bumps

echo "=== nvidia-cuda pip wheels present? (these break svulkan2) ==="
pip list 2>/dev/null | grep -iE "nvidia-cuda|nvidia-cudnn|^jax|tensorflow" || echo "(none — good)"

echo "=== SLIM SETUP DONE -> SimplerEnv render test ==="
export VK_ICD_FILENAMES=$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:$PYTHONPATH
python -c "
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
env=simpler_env.make('google_robot_pick_coke_can'); obs,_=env.reset()
img=get_image_from_maniskill2_obs_dict(env,obs)
print('SLIM_SIMPLER_RENDER_OK', img.shape)
" 2>&1 | tail -10
grep -q SLIM_SIMPLER_RENDER_OK /root/slim_setup.log && touch /root/SLIM_RENDER_OK || touch /root/SLIM_RENDER_FAIL
echo DONE_MARKER
