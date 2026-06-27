# Reproducible frozen-vs-fine-tuned-vs-scratch control (DTU GBar / LSF)

Goal: make the paper's central claim **causal** — show that *frozen ImageNet
features* (not the ResNet architecture, not end-to-end training) drive the
cross-corruption generalization of the safety monitor — by running the matched
control the AIAA abstract promised: the **same ResNet-18**, identical data/folds,
trained three ways (frozen / end-to-end fine-tuned / from-scratch).

## Why the previous attempts failed (and what this fixes)

The repo's "verified" `setup_nvvulkan.sh` has **bit-rotted**: with no lockfile,
pip now resolves deps (JAX/TF/`peft`/`diffusers` → `nvidia-cuda-*` wheels) that
shadow system CUDA and **segfault SAPIEN's `svulkan2` renderer**. Also: NVIDIA
driver 560 hosts segfault even basic SAPIEN render (driver 535 / A40 works), and
the collected frame datasets were ephemeral and got lost with the cloud instances.

Fixes baked into this design:
1. **Slim, pinned env** — no JAX/TF/Octo; `peft`/`diffusers` via `--no-deps`;
   `sdpa` instead of flash-attn; `msgpack`+`rich` for the policy server.
2. **A40 queue (`gpua40`)** — the GPU family that renders (matches the working
   Vast driver-535 case).
3. **Decouple + persist.** Stage A (sim) is run once and its frame dataset is
   persisted off the ephemeral compute node. Stage B (the actual control) is
   **sim-free** and reproduces anywhere.

## Two stages

### Stage A — collect frames (sim-dependent, run once)
`stage_a_collect.py` runs InternVLA-M1 episodes per corruption × budget and saves
`{type}.npz` + `{type}.json`. Kept **torch-free in-process** (talks to the policy
server over a socket) to avoid the torch-CUDA-vs-Vulkan segfault.
**Persist the output** (gbar has no `/work3`; home is ~12 GB free) — either keep
the ~1-2 GB dataset in `~/loo_data` or pull it to the Mac via
`transfer.gbar.dtu.dk`. This dataset is the durable artifact.

### Stage B — the causal control (sim-free, reproducible anywhere)
`stage_b_offline_loo.py` runs leave-one-corruption-out three ways
(`frozen`/`finetune`/`scratch`), identical data/folds/labels/head/epochs — only
the backbone-training treatment changes. Outputs mean ρ and AUROC per treatment.
No SAPIEN, no InternVLA: just trains a ResNet head/backbone on the collected
frames and scores held-out frames.

## GBar specifics (from ~/.gbar.md)
- Scheduler: **LSF** (`#BSUB`, `bsub < job.sh`), not SLURM. Queue: `gpua40`.
- `#BSUB -gpu "num=1:mode=exclusive_process"` is mandatory. `#BSUB -u /dev/null`
  to suppress completion email.
- Everything heavy in `/tmp/$LSB_JOBID`; set `PIP_CACHE_DIR`, `HF_HOME`,
  `TORCH_HOME`, etc. there. `~/.hf_token` exists for the checkpoint.
- Python: `module load python3/3.11.4`. Env tool: `pixi` (no conda/apptainer).

## Status / runbook
1. [in progress] **Render smoke** (`sapien_smoke.bsub`, job 28790638) — validates
   SAPIEN headless render on a gbar A40 before committing to Stage A.
2. [after smoke passes] Finalize `stage_a_collect.bsub` with the smoke-validated
   env/VK_ICD lines; submit; persist `loo_data/`.
3. [after Stage A] Submit Stage B (frozen/finetune/scratch); read the comparison.

Expected headline: frozen mean ρ ≈ 0.79 (matches existing `loo_analysis_v3`);
the new, load-bearing numbers are fine-tuned and from-scratch under identical
conditions. If frozen > fine-tuned > scratch, the causal claim is earned.
