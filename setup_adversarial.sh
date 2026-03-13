#!/bin/bash
# Setup script for adversarial zero-sum dust training on Vast.ai GPU.
# Base image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel (PyTorch already installed)
set -e

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq libvulkan1 mesa-vulkan-drivers libgl1-mesa-glx libglib2.0-0 libegl1-mesa ffmpeg > /dev/null 2>&1

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
pip install "transformers==4.34.0" "setuptools<71" dacite pyyaml "opencv-python-headless<4.11" cma matplotlib imageio pytest einops 2>&1 | tail -1

echo "=== Setting up Vulkan ICD for SAPIEN rendering ==="
apt-get install -y -qq vulkan-tools > /dev/null 2>&1 || true
# Ensure NVIDIA Vulkan ICD is discoverable (some instances have it in /etc/ not /usr/share/)
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
echo 'export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json' >> /root/.bashrc

echo "=== Verifying imports ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}'); import jax; print(f'JAX {jax.__version__}')"

echo "=== Setup complete ==="
echo "Run: cd /root/project && VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json python -m adversarial_dust.run_adversarial --config configs/adversarial_quick_test.yaml"
