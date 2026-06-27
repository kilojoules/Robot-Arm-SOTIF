#!/usr/bin/env python3
"""Two-panel figure for the causal proxy: (1) cross-corruption AUROC vs training
data for frozen/finetune/scratch (the crossover), (2) retained ImageNet AUC vs
data (the feature-drift / forgetting trade-off). Reads results/causal_proxy/
causal_sweep_n*.json and writes crossover.png there."""
import json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = Path("results/causal_proxy")
files = sorted(DIR.glob("causal_sweep_n*.json"),
               key=lambda p: int(re.search(r"n(\d+)\.", p.name).group(1)))
ns, D = [], {t: {"auc": [], "aucs": [], "img": [], "imgs": []}
             for t in ("frozen", "finetune", "scratch")}
baseline = None
for f in files:
    ns.append(int(re.search(r"n(\d+)\.", f.name).group(1)))
    d = json.loads(f.read_text())
    baseline = d.get("imagenet_baseline_auc", baseline)
    for t in D:
        D[t]["auc"].append(d[t]["auroc_mean"]);  D[t]["aucs"].append(d[t]["auroc_std"])
        D[t]["img"].append(d[t]["imagenet_auc_mean"]); D[t]["imgs"].append(d[t]["imagenet_auc_std"])

colors = {"frozen": "C0", "finetune": "C1", "scratch": "C2"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for t in D:
    ax[0].errorbar(ns, D[t]["auc"], yerr=D[t]["aucs"], marker="o", capsize=3, label=t, color=colors[t])
    ax[1].errorbar(ns, D[t]["img"], yerr=D[t]["imgs"], marker="o", capsize=3, label=t, color=colors[t])
ax[0].set(xscale="log", xlabel="training images (n_train)",
          ylabel="held-out corruption AUROC", title="Cross-corruption transfer vs. data")
ax[0].legend(); ax[0].grid(alpha=.3)
if baseline:
    ax[1].axhline(baseline, ls="--", c="gray", label=f"pretrained baseline ({baseline:.3f})")
ax[1].axhline(0.5, ls=":", c="lightgray", label="chance")
ax[1].set(xscale="log", xlabel="training images (n_train)",
          ylabel="retained ImageNet AUC (re-attached fc)", title="Representation preservation vs. data")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout()
out = DIR / "crossover.png"
plt.savefig(out, dpi=200)
print("wrote", out)
