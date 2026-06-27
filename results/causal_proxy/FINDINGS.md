# Causal control: frozen vs fine-tuned vs from-scratch (corruption-transfer proxy)

**Why this exists.** The paper's thesis was "frozen ImageNet features drive the
monitor's cross-corruption generalization." The control needed to make that
*causal* — the same ResNet-18 trained end-to-end vs frozen, under identical
conditions — could not be re-run on the robot sim (SAPIEN 2.2.2 won't render on
current NVIDIA drivers; the original frame data was lost). This is a decoupled,
sim-free proxy that tests the same representation-learning hypothesis: it applies
the paper's own 9 corruption families to standard images (STL-10) and runs the
same leave-one-corruption-out protocol with the same ResNet-18 + tiny head and
the same metrics, for three treatments, across 3 seeds and 4 data sizes.

Plus a feature-drift measure: after training, the **original ImageNet fc is
re-attached** to each backbone and evaluated on ImageNette (macro-OVR AUC) —
quantifying how far each treatment moves the representation.

## Result (mean ± std over 3 seeds; gbar job 28795265)

| n_train | frozen AUROC | finetune AUROC | scratch AUROC | finetune−frozen | finetune ImageNet-AUC (drift) |
|--------:|:------------:|:--------------:|:-------------:|:---------------:|:-----------------------------:|
|   30 | 0.713 ± 0.004 | 0.702 ± 0.013 | 0.604 ± 0.029 | **−0.011** | 0.975 |
|  100 | 0.723 ± 0.002 | 0.761 ± 0.016 | 0.569 ± 0.035 | +0.038 | 0.765 |
|  300 | 0.736 ± 0.005 | 0.814 ± 0.007 | 0.651 ± 0.010 | +0.078 | 0.624 |
| 1000 | 0.731 ± 0.002 | 0.888 ± 0.008 | 0.690 ± 0.022 | +0.157 | 0.538 |

frozen ImageNet-AUC ≈ 0.997 at every size (zero drift, by construction);
scratch ImageNet-AUC ≈ 0.50 (chance) at every size; pretrained baseline = 0.998.
Figure: `crossover.png`.

## What it shows (honest)

1. **Pretraining is essential.** Scratch is worst everywhere and its backbone is
   provably ≈ random (ImageNet AUC ≈ 0.50). Its weakness is data/feature-poverty,
   not just undertraining.
2. **The frozen-vs-fine-tune gap is a function of data.** Fine-tuning's advantage
   grows monotonically with data (−0.011 → +0.157); it vanishes (slightly reverses)
   in the low-data regime, with a crossover around n ≈ 50–100.
3. **Fine-tuning buys detection by forgetting.** Its retained ImageNet AUC falls
   0.975 → 0.538 as data grows, while frozen holds ≈ 0.997. The detection gain is
   paid for with catastrophic forgetting of the general representation.

## The claim this supports (scoped, defensible)

> Pretrained features are necessary for cross-corruption transfer. In the
> data-limited regime characteristic of runtime safety monitoring, freezing them
> matches or beats end-to-end fine-tuning **and** fully preserves the general
> representation; fine-tuning only overtakes given substantially more data, at the
> cost of catastrophic forgetting. The robot monitor (frozen, scarce data,
> ρ=0.79) sits where this curve predicts.

It does **not** support the stronger original wording ("frozen features, not
fine-tuning, are what generalize") — fine-tuning generalizes better given enough
data. The honest, sharper claim is the data-regime one above.

## Caveats

- Proxy task = corruption detection on STL-10, not robot-failure prediction.
- The n=30 frozen-win is ~1σ (marginal); the robust result is the monotone trend.
- Severity-ranking ρ is noisy (n=5 budget points/fold); AUROC is the reliable metric.

## Reproduce (no robot sim, any GPU)

`gbar/stage_c_corruption_transfer.py` (LOO + treatments + seeds + drift);
sweep via `--n-train`; figure via `gbar/plot_crossover.py`. The full sweep is
`gbar/stage_c_sweep.bsub` (LSF, gpua100). Raw outputs: `causal_sweep_n*.json`,
`causal_proxy_seeds.*.json`, `causal_proxy_drift.*.json`.
