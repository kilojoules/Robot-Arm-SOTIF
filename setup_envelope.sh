#!/bin/bash
# Setup script for safe operating envelope predictor on Vast.ai GPU.
# Base image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
# Installs: SimplerEnv, ManiSkill2, Octo, InternVLA-M1, adversarial_dust framework
set -e

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq libvulkan1 mesa-vulkan-drivers libgl1-mesa-glx libglib2.0-0 libegl1-mesa ffmpeg vulkan-tools > /dev/null 2>&1

echo "=== Downgrading numpy for JAX compat ==="
pip install "numpy<2" 2>&1 | tail -1

echo "=== Cloning SimplerEnv ==="
cd /root
if [ ! -d SimplerEnv ]; then
    git clone https://github.com/simpler-env/SimplerEnv
    pip install -e SimplerEnv/ 2>&1 | tail -1
fi

echo "=== Cloning ManiSkill2_real2sim ==="
if [ ! -d ManiSkill2_real2sim ]; then
    git clone https://github.com/simpler-env/ManiSkill2_real2sim
    pip install -e ManiSkill2_real2sim/ 2>&1 | tail -1
fi

echo "=== Cloning Octo (pinned commit) ==="
if [ ! -d octo ]; then
    git clone https://github.com/octo-models/octo
    cd octo && git checkout 653c54acde686fde619855f2eac0dd6edad7116b && pip install -e . 2>&1 | tail -1
    cd /root
fi

echo "=== Patching Octo for JAX 0.4.30 (KeyArray removed) ==="
sed -i 's/PRNGKey = jax.random.KeyArray/PRNGKey = jax.Array/' /root/octo/octo/utils/typing.py

echo "=== Installing JAX ecosystem ==="
pip install "jax[cuda12]==0.4.30" "jaxlib==0.4.30" "flax==0.8.5" "tensorflow==2.15.0" "ml-dtypes==0.2.0" "distrax==0.1.5" "chex==0.1.86" "optax==0.2.3" "orbax-checkpoint==0.4.4" "tensorstore==0.1.45" "tensorflow-probability==0.23.0" 2>&1 | tail -1

echo "=== Installing remaining deps ==="
pip install "transformers==4.34.0" "setuptools<71" dacite pyyaml "opencv-python-headless<4.11" cma matplotlib imageio pytest einops websockets qwen-vl-utils omegaconf accelerate timm transforms3d 2>&1 | tail -1

echo "=== Setting up InternVLA-M1 ==="
cd /root
if [ ! -d InternVLA-M1 ]; then
    git clone https://github.com/InternRobotics/InternVLA-M1
fi

echo "=== Installing InternVLA-M1 deps ==="
pip install "transformers==4.52.3" "flash-attn==2.7.3" --no-build-isolation 2>&1 | tail -1 || echo "flash-attn install may require manual build"
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1 || echo "Using existing PyTorch"

echo "=== Downloading InternVLA-M1 checkpoint ==="
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
print('Checkpoint downloaded successfully')
"
fi

echo "=== Cloning camera_occlusion (envelope-predictor branch) ==="
cd /root
if [ ! -d project ]; then
    git clone -b feature/envelope-predictor https://github.com/kilojoules/camera_occlusion project
fi

echo "=== Setting up Vulkan ICD ==="
# Path varies by machine — find the actual nvidia ICD file
VK_ICD=$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
if [ -n "$VK_ICD" ]; then
    export VK_ICD_FILENAMES="$VK_ICD"
    echo "export VK_ICD_FILENAMES=$VK_ICD" >> /root/.bashrc
    echo "Vulkan ICD: $VK_ICD"
else
    echo "WARNING: nvidia_icd.json not found"
fi

echo "=== Verifying imports ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}'); import jax; print(f'JAX {jax.__version__}')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start InternVLA-M1 server:"
echo "  cd /root/InternVLA-M1 && python deployment/model_server/server_policy_M1.py --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt --port 10093 --use_bf16 &"
echo ""
echo "To run envelope predictor:"
echo "  cd /root/project && VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json python -m adversarial_dust.run_envelope --config configs/envelope_eggplant_internvla.yaml"
echo ""
echo "For quick test (octo-small, no InternVLA server needed):"
echo "  cd /root/project && VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json python -m adversarial_dust.run_envelope --config configs/envelope_quick_test.yaml"
