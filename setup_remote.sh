#!/bin/bash
# Setup script for running adversarial dust experiments on a remote GPU machine.
# This installs SimplerEnv, Octo, and our adversarial_dust package.
set -euo pipefail

echo "=== Setting up adversarial dust experiment ==="

# System dependencies
apt-get update -qq && apt-get install -y -qq git wget ffmpeg libgl1-mesa-glx libglib2.0-0 libegl1-mesa libgles2-mesa > /dev/null 2>&1
echo "System deps installed."

# Clone SimplerEnv (includes ManiSkill2 setup)
if [ ! -d "/workspace/SimplerEnv" ]; then
    cd /workspace
    git clone https://github.com/simpler-env/SimplerEnv.git
    cd SimplerEnv
    pip install -e . 2>&1 | tail -3
    echo "SimplerEnv installed."
else
    echo "SimplerEnv already exists."
fi

# Install ManiSkill2_real2sim (required by SimplerEnv)
if [ ! -d "/workspace/ManiSkill2_real2sim" ]; then
    cd /workspace
    git clone https://github.com/simpler-env/ManiSkill2_real2sim.git
    cd ManiSkill2_real2sim
    pip install -e . 2>&1 | tail -3
    echo "ManiSkill2_real2sim installed."
else
    echo "ManiSkill2_real2sim already exists."
fi

# Install Octo
pip install octo 2>&1 | tail -3 || {
    echo "pip install octo failed, trying from source..."
    if [ ! -d "/workspace/octo" ]; then
        cd /workspace
        git clone https://github.com/octo-models/octo.git
        cd octo
        pip install -e . 2>&1 | tail -3
    fi
}
echo "Octo installed."

# Install our project dependencies
pip install dacite pyyaml opencv-python-headless cma matplotlib imageio 2>&1 | tail -3
echo "Project dependencies installed."

# Copy our project code
echo "Project code should be at /workspace/adversarial_dust_project/"

echo "=== Setup complete ==="
echo "Run with: cd /workspace/adversarial_dust_project && python -m adversarial_dust.run_experiment --config configs/quick_test.yaml"
