#!/bin/bash
set -e

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq libvulkan1 mesa-vulkan-drivers libgl1-mesa-glx libglib2.0-0 libegl1-mesa ffmpeg > /dev/null 2>&1

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

echo "=== Cloning Octo ==="
if [ ! -d octo ]; then
    git clone https://github.com/octo-models/octo
    cd octo && git checkout 653c54acde686fde619855f2eac0dd6edad7116b && pip install -e . 2>&1 | tail -1
    cd /root
fi

echo "=== Installing JAX ecosystem ==="
pip install "jax[cuda12]==0.4.30" "jaxlib==0.4.30" "flax==0.8.5" "tensorflow==2.15.0" "ml-dtypes==0.2.0" "distrax==0.1.5" "chex==0.1.86" "optax==0.2.3" "orbax-checkpoint==0.4.4" "tensorstore==0.1.45" "tensorflow-probability==0.23.0" 2>&1 | tail -1

echo "=== Installing remaining deps ==="
pip install "transformers==4.34.0" "setuptools<71" dacite pyyaml opencv-python-headless cma matplotlib imageio pytest 2>&1 | tail -1

echo "=== Setup complete ==="
