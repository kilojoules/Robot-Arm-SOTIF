# Robot-Arm-SOTIF

**A 33K-parameter visual safety monitor that predicts robot manipulation failure under camera corruption -- and generalizes to corruption types never seen in training.**

[Paper (RA-L, in preparation)]() | [Results](results/loo_analysis/loo_summary.json)

---

## Key Result

A frozen ResNet-18 backbone trained on 8 camera corruption types accurately predicts policy failure on the held-out 9th type. Leave-one-out cross-validation across 9 corruption types spanning 6 physical mechanisms:

<p align="center">
  <img src="docs/figures/loo_results.png" alt="LOO cross-corruption generalization results" width="900">
</p>

Of the 5 corruption types that cause policy failures, the monitor achieves **mean Spearman rho = 0.883 +/- 0.098** for severity ranking and **mean AUROC = 0.915 +/- 0.076** for episode-level failure detection. 4 corruption types are policy-robust (InternVLA-M1 does not fail even at 90% corruption budget).

| Held-out Type | Category | Spearman rho | p-value | AUROC |
|---|---|:---:|:---:|:---:|
| Fingerprint | Lens contact | **0.975** | 0.005 | 0.771 |
| Rain | Lens contact | **0.975** | 0.005 | 0.931 |
| Motion blur | Blur | **0.894** | 0.041 | 0.943 |
| Defocus blur | Optical | **0.866** | 0.058 | 0.929 |
| Low-light | Illumination | **0.707** | 0.182 | 1.000 |

## Corruption Taxonomy

9 corruption types organized by physical mechanism, following [ImageNet-C](https://arxiv.org/abs/1903.12261) adapted for fixed-mount robot cameras:

<p align="center">
  <img src="docs/figures/corruption_grid.png" alt="9 corruption types at 70% budget" width="900">
</p>

| Category | Corruption | Params | Source |
|---|---|:---:|---|
| Lens contact | Fingerprint smudge | 24 | Gu et al., SIGGRAPH Asia 2009 |
| Lens contact | Rain drops | 5 | camera_occlusion |
| Optical | Glare / lens flare | 6 | Physically-inspired |
| Optical | Defocus blur | 1 | ImageNet-C |
| Blur | Motion blur | 2 | ImageNet-C |
| Sensor | Gaussian noise | 2 | ImageNet-C |
| Atmospheric | Fog / haze | 2 | ImageNet-C |
| Illumination | Low-light | 2 | LIBERO-Plus |
| Digital | JPEG compression | 1 | ImageNet-C |

## Per-Fold Analysis

Predicted failure probability tracks actual failure rate across occlusion budgets for all 5 failure-inducing corruption types:

<p align="center">
  <img src="docs/figures/loo_per_fold.png" alt="Per-fold budget vs prediction" width="900">
</p>

## Feature-Space Similarity

Pairwise cosine similarity between corruption types in frozen ResNet-18 feature space:

<p align="center">
  <img src="docs/figures/feature_similarity.png" alt="Feature similarity heatmap" width="800">
</p>

## Architecture

<p align="center">
  <img src="docs/figures/architecture.png" alt="SafetyNet architecture" width="700">
</p>

Frozen ResNet-18 backbone (ImageNet-pretrained, 11M params frozen) + trainable classification head (33K params): `512 -> FC(64) -> ReLU -> Dropout(0.3) -> FC(1) -> sigmoid`. Trained with class-weighted BCE loss and episode-level train/val split.

## Method

1. **Adversarial CMA-ES** finds worst-case occlusion patterns for each corruption type and budget level
2. **VLM policy** ([InternVLA-M1](https://github.com/OpenGVLab/InternVLA)) executes pick-and-place tasks in simulation ([SimplerEnv](https://github.com/simpler-env/SimplerEnv) / SAPIEN)
3. **Safety predictor** is trained on (image, success/failure) pairs from these episodes
4. **Leave-one-out evaluation**: train on 8 corruption types, evaluate on the held-out 9th

## Reproducing Results

### Requirements

- GPU machine with CUDA (experiments run on [vast.ai](https://vast.ai) RTX 4090 instances)
- [pixi](https://pixi.sh) for local dependency management

### Full LOO pipeline (GPU required)

<details>
<summary>GPU setup and execution</summary>

```bash
# On vast.ai with nvidia/vulkan:1.3-470 image:
bash scripts/setup_nvvulkan.sh

# Start InternVLA-M1 server
cd /root/InternVLA-M1
PYTHONPATH=/root/InternVLA-M1 python deployment/model_server/server_policy_M1.py \
  --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
  --port 10093 --use_bf16 &
sleep 60

# Run LOO analysis
cd /root/project
PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion \
  python -u adversarial_dust/run_safety_predictor.py \
  --config configs/safety_predictor.yaml \
  --loo \
  --loo-types fingerprint glare rain gaussian_noise jpeg \
             motion_blur defocus_blur fog low_light \
  --eval-episodes 10 \
  --episodes-per-condition 10 \
  --output-dir results/loo_analysis
```
</details>

### Generate figures locally

```bash
pixi run python scripts/generate_paper_figures.py \
  --results-dir results/loo_analysis \
  --output-dir docs/figures
```

## SOTIF Context

[SOTIF (ISO 21448)](https://www.iso.org/standard/77490.html) addresses safety of the intended functionality -- failures that arise not from hardware faults, but from limitations of perception and decision-making under real-world conditions. Camera corruption is a canonical SOTIF triggering condition.

This project demonstrates that a lightweight visual safety monitor can detect degraded perception and **generalize across corruption types** using pretrained visual features, reducing the need for corruption-specific safety validation.

## Citation

```bibtex
@article{quick2026crosscorruption,
  title={Cross-Corruption Safety Monitoring for Vision-Based Robot Manipulation
         via Adversarial Envelope Analysis},
  author={Quick, Julian},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  note={In preparation}
}
```
