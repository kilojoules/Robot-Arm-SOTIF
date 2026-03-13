#!/bin/bash
# Verified setup script for nvidia/vulkan:1.3-470 on Vast.ai
# Installs SimplerEnv + InternVLA-M1 + adversarial_dust framework
# Tested 2026-03-13 — the ONLY combo that actually works.
set -e

echo "=== System dependencies ==="
apt-get update -qq
apt-get install -y -qq git wget libgl1-mesa-glx libglib2.0-0 ffmpeg cmake build-essential tmux > /dev/null 2>&1

echo "=== Installing Miniconda (Python 3.11) ==="
if [ ! -d /opt/miniconda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    rm /tmp/miniconda.sh
fi
export PATH="/opt/miniconda/bin:$PATH"
echo 'export PATH="/opt/miniconda/bin:$PATH"' >> /root/.bashrc

echo "=== PyTorch ==="
pip install torch==2.5.1+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip install "numpy<2" 2>&1 | tail -1

echo "=== SAPIEN + Gymnasium ==="
pip install sapien==2.2.2 gymnasium ruckig 2>&1 | tail -1

echo "=== SimplerEnv ==="
cd /root
[ ! -d SimplerEnv ] && git clone https://github.com/simpler-env/SimplerEnv
pip install -e SimplerEnv/ 2>&1 | tail -1

echo "=== ManiSkill2 ==="
[ ! -d ManiSkill2_real2sim ] && git clone https://github.com/simpler-env/ManiSkill2_real2sim
pip install -e ManiSkill2_real2sim/ 2>&1 | tail -1

echo "=== Octo (pinned) ==="
if [ ! -d octo ]; then
    git clone https://github.com/octo-models/octo
    cd octo && git checkout 653c54acde686fde619855f2eac0dd6edad7116b
    pip install -e . 2>&1 | tail -1
    cd /root
fi
sed -i 's/PRNGKey = jax.random.KeyArray/PRNGKey = jax.Array/' /root/octo/octo/utils/typing.py 2>/dev/null || true

echo "=== JAX ecosystem ==="
pip install "jax[cuda12]==0.4.30" "jaxlib==0.4.30" "flax==0.8.5" "tensorflow==2.15.0" \
    "ml-dtypes==0.2.0" "distrax==0.1.5" "chex==0.1.86" "optax==0.2.3" \
    "orbax-checkpoint==0.4.4" "tensorstore==0.1.45" "tensorflow-probability==0.23.0" 2>&1 | tail -1

echo "=== InternVLA-M1 ==="
cd /root
[ ! -d InternVLA-M1 ] && git clone https://github.com/InternRobotics/InternVLA-M1

echo "=== CUDA toolkit for flash-attn build ==="
conda install -y -c nvidia cuda-toolkit=12.4 2>&1 | tail -3

echo "=== flash-attn ==="
pip install flash-attn==2.7.3 --no-build-isolation 2>&1 | tail -1

echo "=== Python deps ==="
pip install "transformers==4.52.3" dacite pyyaml cma matplotlib imageio \
    einops websockets qwen-vl-utils omegaconf accelerate timm transforms3d \
    "setuptools<71" pytest 2>&1 | tail -1

echo "=== InternVLA-M1 checkpoint ==="
if [ ! -f /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt ]; then
    pip install huggingface-hub 2>&1 | tail -1
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
    git clone -b feature/envelope-predictor https://github.com/kilojoules/camera_occlusion
    pip install -e camera_occlusion/ 2>&1 | tail -1
fi

echo "=== Robot-Arm-SOTIF project ==="
cd /root
if [ ! -d project ]; then
    git clone https://github.com/kilojoules/Robot-Arm-SOTIF project
fi

echo "=== Post-install fixes ==="
# numpy gets bumped to 2.x by dependencies — force back
pip install "numpy<2" "opencv-python-headless<4.11" 2>&1 | tail -1
# transformers can get uninstalled — reinstall
pip install "transformers==4.52.3" 2>&1 | tail -1
# triton needs libcuda.so symlink
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true

echo "=== Vulkan ICD ==="
VK_ICD=$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
if [ -n "$VK_ICD" ]; then
    export VK_ICD_FILENAMES="$VK_ICD"
    echo "export VK_ICD_FILENAMES=$VK_ICD" >> /root/.bashrc
    echo "Vulkan ICD: $VK_ICD"
fi

echo "=== Verify ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup complete ==="
echo "Start InternVLA-M1 server:"
echo "  cd /root/InternVLA-M1 && PYTHONPATH=/root/InternVLA-M1 python deployment/model_server/server_policy_M1.py --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt --port 10093 --use_bf16 &"
echo ""
echo "Run safety predictor pipeline:"
echo "  cd /root/project && PYTHONPATH=/root/InternVLA-M1:/root/project python -u adversarial_dust/run_safety_predictor.py --config configs/safety_predictor.yaml"
