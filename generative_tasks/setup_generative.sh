#!/bin/bash
# Setup script for generative_tasks pipeline on Vast.ai
# Generates 3D meshes (ice cream, shrimp, chalice, bucket) using Shap-E,
# converts to URDF, and builds task matrix.
set -e

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y git wget

echo "=== Installing Python dependencies ==="
pip install trimesh numpy Pillow

echo "=== Installing Shap-E ==="
pip install git+https://github.com/openai/shap-e.git

echo "=== Cloning generative_tasks code ==="
mkdir -p /root/generative_tasks
# Files will be copied via scp before this script runs

echo "=== Running mesh generation pipeline ==="
cd /root/generative_tasks
python run_pipeline.py --num-steps 64 --guidance-scale 15.0

echo "=== Pipeline complete ==="
echo "Generated assets in /root/generative_tasks/assets/"
echo "URDF bundles in /root/generative_tasks/urdf_assets/"
echo "Task specs in /root/generative_tasks/task_specs.json"
ls -la /root/generative_tasks/assets/objects/
ls -la /root/generative_tasks/assets/containers/
ls -la /root/generative_tasks/urdf_assets/ 2>/dev/null || echo "No URDF output yet"
