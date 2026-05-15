#!/bin/bash
# Setup script for pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel on Vast.ai.
#
# Alternative to setup_nvvulkan.sh — switched after the nvidia/vulkan:1.3-470
# image preempted twice within ~50 min of LOO start. The hypothesis is that
# the community Vulkan image triggers some host watchdog; the official
# PyTorch image is more thoroughly tested.
#
# Installs the Vulkan stack on top of the pytorch image (which lacks libvulkan
# entirely), then SimplerEnv + InternVLA-M1 + adversarial_dust.
set -e

# NVIDIA repo signing key may be missing on fresh images
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub > /dev/null 2>&1 || true
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub > /dev/null 2>&1 || true

echo "=== System dependencies ==="
apt-get update -qq
apt-get install -y -qq git wget curl libgl1-mesa-glx libglib2.0-0 ffmpeg \
    cmake build-essential tmux mesa-vulkan-drivers \
    libvulkan1 libvulkan-dev vulkan-tools > /dev/null 2>&1 || \
apt-get install -y -qq git wget curl libgl1-mesa-glx libglib2.0-0 ffmpeg \
    cmake build-essential tmux > /dev/null 2>&1

echo "=== Vulkan 1.3 SDK (overrides system libvulkan if too old) ==="
if [ ! -d /opt/vulkan/1.3.290.0 ]; then
    mkdir -p /opt/vulkan
    curl -s -L "https://sdk.lunarg.com/sdk/download/1.3.290.0/linux/vulkansdk-linux-x86_64-1.3.290.0.tar.xz" -o /tmp/vksdk.tar.xz
    tar -xJf /tmp/vksdk.tar.xz -C /opt/vulkan
    rm /tmp/vksdk.tar.xz
fi
export LD_LIBRARY_PATH="/opt/vulkan/1.3.290.0/x86_64/lib:$LD_LIBRARY_PATH"
grep -q '1.3.290.0/x86_64/lib' /root/.bashrc || \
    echo 'export LD_LIBRARY_PATH="/opt/vulkan/1.3.290.0/x86_64/lib:$LD_LIBRARY_PATH"' >> /root/.bashrc

echo "=== NVIDIA Vulkan ICD JSON (point loader at host-mounted driver) ==="
# Vast.ai mounts the host's NVIDIA driver into the container; the Vulkan
# loader needs an ICD manifest pointing at libGLX_nvidia.so.0. Write it
# to /root/ since the pytorch image has /etc/vulkan/ mounted read-only.
# Prefer the system ICD if one already exists (nvidia/vulkan image).
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    echo "Using existing /etc/vulkan/icd.d/nvidia_icd.json"
    ICD_PATH=/etc/vulkan/icd.d/nvidia_icd.json
else
    cat > /root/nvidia_icd.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.250"
    }
}
EOF
    ICD_PATH=/root/nvidia_icd.json
fi
export VK_ICD_FILENAMES="$ICD_PATH"
grep -q VK_ICD_FILENAMES /root/.bashrc || \
    echo "export VK_ICD_FILENAMES=$ICD_PATH" >> /root/.bashrc

echo "=== Verify Vulkan sees the GPU ==="
LD_LIBRARY_PATH="/opt/vulkan/1.3.290.0/x86_64/lib:$LD_LIBRARY_PATH" \
    /opt/vulkan/1.3.290.0/x86_64/bin/vulkaninfo 2>&1 | grep -E 'deviceName|driverInfo' | head -3

echo "=== Conda environment (use existing pytorch image conda) ==="
# pytorch image uses /opt/conda. Put it on PATH if not already.
if [ -d /opt/conda ]; then
    export PATH="/opt/conda/bin:$PATH"
    grep -q '/opt/conda/bin' /root/.bashrc || \
        echo 'export PATH="/opt/conda/bin:$PATH"' >> /root/.bashrc
elif [ -d /opt/miniconda ]; then
    export PATH="/opt/miniconda/bin:$PATH"
fi
which python && python --version
python -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())'

echo "=== SAPIEN + Gymnasium ==="
pip install --quiet sapien==2.2.2 gymnasium ruckig 2>&1 | tail -1

echo "=== SimplerEnv ==="
cd /root
[ ! -d SimplerEnv ] && git clone -q https://github.com/simpler-env/SimplerEnv
pip install --quiet -e SimplerEnv/ 2>&1 | tail -1

echo "=== ManiSkill2_real2sim ==="
cd /root
[ ! -d ManiSkill2_real2sim ] && git clone -q https://github.com/simpler-env/ManiSkill2_real2sim
pip install --quiet -e ManiSkill2_real2sim/ 2>&1 | tail -1

echo "=== Octo (pinned) ==="
cd /root
[ ! -d octo ] && git clone -q https://github.com/octo-models/octo && \
    cd octo && git checkout -q 653c54acde686fde619855f2eac0dd6edad7116b && cd ..
pip install --quiet -e /root/octo 2>&1 | tail -1

echo "=== JAX ecosystem ==="
pip install --quiet "jax[cuda12]==0.4.34" "flax==0.10.0" \
    "orbax-checkpoint==0.4.4" "tensorstore==0.1.45" "tensorflow-probability==0.23.0" 2>&1 | tail -1

echo "=== InternVLA-M1 ==="
cd /root
[ ! -d InternVLA-M1 ] && git clone -q https://github.com/InternRobotics/InternVLA-M1

echo "=== CUDA toolkit (for flash-attn build) ==="
# pytorch image already has CUDA toolkit at /usr/local/cuda; flash-attn needs nvcc
which nvcc || conda install -y -q -c nvidia 'cuda-toolkit=12.4' 2>&1 | tail -1

echo "=== flash-attn ==="
pip install --quiet flash-attn --no-build-isolation 2>&1 | tail -1

echo "=== Python deps ==="
pip install --quiet "transformers==4.52.3" dacite pyyaml cma matplotlib imageio \
    einops websockets qwen-vl-utils omegaconf accelerate timm transforms3d \
    "setuptools<71" pytest 2>&1 | tail -1

echo "=== InternVLA-M1 checkpoint ==="
if [ ! -f /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt ]; then
    pip install --quiet huggingface-hub 2>&1 | tail -1
    python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('/root/internvla_m1_ckpt/checkpoints', exist_ok=True)
for f in ['checkpoints/steps_50000_pytorch_model.pt', 'config.yaml', 'dataset_statistics.json']:
    print(f'Downloading {f}...')
    hf_hub_download(
        repo_id='InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge',
        filename=f,
        local_dir='/root/internvla_m1_ckpt',
    )
print('Checkpoint downloaded.')
"
fi

echo "=== camera_occlusion (for Rain model) ==="
cd /root
if [ ! -d camera_occlusion ]; then
    git clone -q -b feature/envelope-predictor https://github.com/kilojoules/camera_occlusion
    pip install --quiet -e camera_occlusion/ 2>&1 | tail -1
fi

echo "=== Robot-Arm-SOTIF project ==="
cd /root
if [ ! -d project ]; then
    git clone -q https://github.com/kilojoules/Robot-Arm-SOTIF project
fi

echo "=== Post-install fixes ==="
pip install --quiet "numpy<2" "opencv-python-headless<4.11" 2>&1 | tail -1
pip install --quiet "transformers==4.52.3" 2>&1 | tail -1
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true

echo "=== Verify ==="
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
LD_LIBRARY_PATH="/opt/vulkan/1.3.290.0/x86_64/lib:$LD_LIBRARY_PATH" \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    python -c "import sapien.core as sc; r = sc.SapienRenderer(offscreen_only=True); print('SAPIEN renderer OK')"

echo "=== Setup complete ==="
echo "Start InternVLA-M1 server:"
echo "  cd /root/InternVLA-M1 && python deployment/model_server/server_policy_M1.py --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt --port 10093 --use_bf16 &"
