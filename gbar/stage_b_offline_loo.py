#!/usr/bin/env python3
"""Stage B: the causal control, fully OFFLINE (no robot sim).

Given the per-corruption frame datasets collected in Stage A
(``{type}.npz`` with images/labels + ``{type}.json`` with per-frame metadata),
run the SAME leave-one-corruption-out protocol three ways with everything
else held identical:

    frozen   : ResNet-18, ImageNet weights, backbone frozen (only head trains)
    finetune : ResNet-18, ImageNet weights, end-to-end fine-tuned (backbone_lr=lr/10)
    scratch  : ResNet-18, random init, end-to-end trained

Only the backbone-training treatment changes; the data, folds, labels, head,
epochs, and eval are identical. That is the matched control the AIAA abstract
promised ("the same ResNet architecture but with all the parameters free to be
tuned under identical conditions") and the comparison that lets us state the
causal claim (frozen *features* — not architecture, not end-to-end training —
drive cross-corruption transfer).

No SAPIEN, no InternVLA, no policy server: this trains/evaluates a classifier
on pre-collected frames, so it reproduces anywhere with a GPU.

Usage:
    python gbar/stage_b_offline_loo.py \
        --data-dir /path/to/loo_data \
        --out results/stage_b_causal/summary.json \
        --epochs 50 --device cuda
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from adversarial_dust.safety_predictor import train_safety_predictor

TREATMENTS = {
    # name        (freeze_backbone, pretrained)
    "frozen":   (True,  True),
    "finetune": (False, True),
    "scratch":  (False, False),
}
DEFAULT_TYPES = ["fingerprint", "glare", "rain", "gaussian_noise", "jpeg",
                 "motion_blur", "defocus_blur", "dust_camera", "low_light"]


def load_type(data_dir: Path, t: str):
    """Return (images[N,H,W,3] uint8, samples[list of per-frame meta dicts])."""
    npz = np.load(str(data_dir / f"{t}.npz"))
    meta = json.loads((data_dir / f"{t}.json").read_text())
    samples = meta["samples"]
    images = npz["images"]
    assert len(samples) == len(images), f"{t}: meta/image length mismatch"
    return images, samples


def offline_eval(predictor, images, samples, batch=256):
    """Score held-out frames, aggregate to episodes, return (rho, auroc, n_ep).

    rho   : Spearman(empirical failure rate, mean predicted) over budget levels.
    auroc : episode-level AUROC (predicted mean p_fail vs actual failure).
    Clean (budget==0) frames are excluded, matching the LOO eval protocol.
    """
    preds = []
    for i in range(0, len(images), batch):
        preds.append(predictor.predict_batch(list(images[i:i + batch])))
    p_fail = np.concatenate(preds)

    # group frames into episodes keyed by (budget, episode), excluding clean
    episodes = {}
    for i, s in enumerate(samples):
        b = float(s["budget"])
        if b <= 0.0:
            continue
        key = (round(b, 4), int(s["episode"]))
        ep = episodes.setdefault(key, {"p": [], "fail": 1 - int(bool(s["success"]))})
        ep["p"].append(p_fail[i])

    ep_pred, ep_fail, ep_budget = [], [], []
    for (b, _epi), ep in episodes.items():
        ep_pred.append(float(np.mean(ep["p"])))
        ep_fail.append(ep["fail"])
        ep_budget.append(b)
    ep_pred, ep_fail, ep_budget = map(np.array, (ep_pred, ep_fail, ep_budget))

    # per-budget empirical failure rate vs mean predicted -> Spearman rho
    budgets = sorted(set(ep_budget.tolist()))
    fail_rate, mean_pred = [], []
    for b in budgets:
        m = ep_budget == b
        fail_rate.append(float(np.mean(ep_fail[m])))
        mean_pred.append(float(np.mean(ep_pred[m])))
    rho = (float(spearmanr(fail_rate, mean_pred)[0])
           if len(set(fail_rate)) > 1 else float("nan"))
    auroc = (float(roc_auc_score(ep_fail, ep_pred))
             if len(set(ep_fail.tolist())) > 1 else float("nan"))
    return rho, auroc, int(len(ep_fail)), budgets, fail_rate, mean_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--types", nargs="+", default=DEFAULT_TYPES)
    ap.add_argument("--treatments", nargs="+", default=list(TREATMENTS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--frames-per-episode", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # one shared cache of each type's frames
    cache = {t: load_type(args.data_dir, t) for t in args.types}
    results = {}

    for treat in args.treatments:
        freeze, pretrained = TREATMENTS[treat]
        folds = {}
        for held_out in args.types:
            train_types = [t for t in args.types if t != held_out]
            tr_imgs = np.concatenate([cache[t][0] for t in train_types])
            tr_lbls = np.concatenate(
                [np.array([1 - int(bool(s["success"])) for s in cache[t][1]],
                          dtype=np.float32) for t in train_types])
            with tempfile.TemporaryDirectory() as td:
                merged = Path(td) / "train.npz"
                np.savez_compressed(str(merged), images=tr_imgs, labels=tr_lbls)
                predictor = train_safety_predictor(
                    dataset_path=str(merged), output_dir=str(Path(td) / "model"),
                    epochs=args.epochs, lr=args.lr, device=args.device,
                    frames_per_episode=args.frames_per_episode,
                    backbone="resnet18", freeze_backbone=freeze,
                    pretrained=pretrained, seed=args.seed)
                imgs, samples = cache[held_out]
                rho, auroc, n_ep, budgets, fr, mp = offline_eval(
                    predictor, imgs, samples)
            folds[held_out] = {"rho": rho, "auroc": auroc, "n_episodes": n_ep,
                               "budgets": budgets, "failure_rate": fr,
                               "mean_pred": mp}
            print(f"[{treat}] held-out {held_out:14s} rho={rho:.3f} "
                  f"auroc={auroc:.3f} (n_ep={n_ep})", flush=True)
        valid_rho = [f["rho"] for f in folds.values() if f["rho"] == f["rho"]]
        valid_auroc = [f["auroc"] for f in folds.values() if f["auroc"] == f["auroc"]]
        results[treat] = {
            "per_fold": folds,
            "mean_rho": float(np.mean(valid_rho)) if valid_rho else float("nan"),
            "mean_auroc": float(np.mean(valid_auroc)) if valid_auroc else float("nan"),
            "n_valid_folds": len(valid_rho),
        }
        print(f"=== {treat}: mean rho={results[treat]['mean_rho']:.3f} "
              f"AUROC={results[treat]['mean_auroc']:.3f} "
              f"over {len(valid_rho)} folds ===", flush=True)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    print("\nCAUSAL COMPARISON (mean over valid folds):")
    for treat in args.treatments:
        r = results[treat]
        print(f"  {treat:9s}  rho={r['mean_rho']:+.3f}  AUROC={r['mean_auroc']:.3f}")


if __name__ == "__main__":
    main()
