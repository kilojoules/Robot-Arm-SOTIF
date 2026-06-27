#!/usr/bin/env python3
"""Causal control via a corruption-transfer proxy (no robot sim, no Vulkan).

SAPIEN 2.2.2 cannot render on current GPU drivers (gbar fleet = driver 595), so
the robot frozen-vs-fine-tuned control can't be re-collected. This decoupled
experiment tests the SAME representation-learning hypothesis on standard images:

    Do frozen ImageNet features let a corruption detector generalize to
    *unseen corruption types* better than fine-tuning or training from scratch?

It reuses the paper's OWN corruption models (so the corruption families match the
robot study) and the SAME ResNet-18 + tiny-head architecture and metrics
(severity-ranking Spearman rho + clean/corrupt AUROC), under a leave-one-
corruption-out protocol, for three treatments:

    frozen   : ResNet-18 ImageNet weights, backbone frozen (head-only training)
    finetune : ResNet-18 ImageNet weights, end-to-end (backbone_lr = lr/10)
    scratch  : ResNet-18 random init, end-to-end

Each treatment is run under several seeds (--seeds) for mean +/- std error bars.

Optional feature-drift measure (--imagenet-dir): after training, re-attach the
original ImageNet fc to each backbone and measure retained ImageNet skill on
ImageNette (quantifies how far fine-tuning/scratch move the representation).

Needs only torch/torchvision + cv2/numpy + the corruption models. Runs on any GPU.

Usage:
    python gbar/stage_c_corruption_transfer.py --out results/causal/summary.json \
        --data-root /tmp/$LSB_JOBID/data --device cuda --n-train 300 --epochs 20 \
        --seeds 42,123,456 --imagenet-dir /tmp/$LSB_JOBID/imagenette2-160/val
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torchvision import models, datasets

from adversarial_dust.config import FingerprintConfig, GlareConfig
from adversarial_dust.digital_corruption import (
    GaussianNoiseModel, JPEGCompressionModel, MotionBlurModel,
    DefocusBlurModel, FogModel, LowLightModel)
from adversarial_dust.dust_camera_model import DustCameraModel
from adversarial_dust.fingerprint_model import FingerprintSmudgeModel
from adversarial_dust.glare_model import AdversarialGlareModel

SHAPE = (224, 224, 3)
BUDGETS = [0.1, 0.3, 0.5, 0.7, 0.9]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
TREATMENTS = {"frozen": (True, True), "finetune": (False, True), "scratch": (False, False)}
CORRUPTIONS = ["fingerprint", "glare", "rain", "gaussian_noise", "jpeg",
               "motion_blur", "defocus_blur", "dust_camera", "low_light"]
# ImageNette = 10 ImageNet classes; their ImageNet-1k indices (wnid alpha order).
IMAGENETTE_IMAGENET_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def make_corruption(name, budget):
    if name == "gaussian_noise": return GaussianNoiseModel(budget_level=budget, image_shape=SHAPE)
    if name == "jpeg":           return JPEGCompressionModel(budget_level=budget, image_shape=SHAPE)
    if name == "motion_blur":    return MotionBlurModel(budget_level=budget, image_shape=SHAPE)
    if name == "defocus_blur":   return DefocusBlurModel(budget_level=budget, image_shape=SHAPE)
    if name == "fog":            return FogModel(budget_level=budget, image_shape=SHAPE)
    if name == "low_light":      return LowLightModel(budget_level=budget, image_shape=SHAPE)
    if name == "dust_camera":    return DustCameraModel(budget_level=budget, image_shape=SHAPE)
    if name == "fingerprint":    return FingerprintSmudgeModel(FingerprintConfig(), SHAPE, budget)
    if name == "glare":          return AdversarialGlareModel(GlareConfig(), SHAPE, budget)
    if name == "rain":
        from adversarial_dust.rain_model import RainOcclusionModel  # needs camera_occlusion
        return RainOcclusionModel(budget_level=budget, image_shape=SHAPE)
    raise ValueError(name)


def corrupt(img, name, budget, rng, ts):
    m = make_corruption(name, budget)
    return m.apply(img, m.get_random_params(rng), timestep=ts)


def preprocess(imgs_uint8):
    x = np.stack(imgs_uint8).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2))


class Net(nn.Module):
    """Same architecture as the paper's monitor: ResNet-18 + 512->64->1 head."""
    def __init__(self, freeze, pretrained):
        super().__init__()
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=w)
        feat = base.fc.in_features
        base.fc = nn.Identity()
        if freeze:
            for p in base.parameters():
                p.requires_grad = False
        self.backbone, self.head = base, nn.Sequential(
            nn.Linear(feat, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.backbone(x))


def load_clean(data_root, n):
    import cv2
    ds = datasets.STL10(root=data_root, split="train", download=True)
    return [cv2.resize(np.array(ds[i][0]), (224, 224)) for i in range(min(n, len(ds)))]


def train(model, X, y, device, epochs, lr):
    model.to(device).train()
    pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    if any(p.requires_grad for p in model.backbone.parameters()):   # finetune / scratch
        opt = torch.optim.Adam([{"params": model.backbone.parameters(), "lr": lr / 10},
                                {"params": model.head.parameters(), "lr": lr}])
    else:                                                            # frozen
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    yt = torch.from_numpy(y.astype(np.float32))
    idx = np.arange(len(X))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]
            xb, yb = X[b].to(device), yt[b].to(device).unsqueeze(1)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()


@torch.no_grad()
def score(model, X, device, bs=128):
    model.eval()
    return np.concatenate([torch.sigmoid(model(X[i:i + bs].to(device))).cpu().numpy().ravel()
                           for i in range(0, len(X), bs)])


def load_imagenette(val_dir, per_class=50):
    import os, glob, cv2
    wnids = sorted(d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
    imgs, labs = [], []
    for ci, w in enumerate(wnids):
        for f in sorted(glob.glob(os.path.join(val_dir, w, "*.JPEG")))[:per_class]:
            im = cv2.imread(f)
            if im is None:
                continue
            imgs.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)))
            labs.append(ci)
    return imgs, np.array(labs)


@torch.no_grad()
def imagenet_competence(backbone, orig_fc, X, y, device, bs=128):
    """Glue the original ImageNet fc onto `backbone`; macro-OVR AUC on ImageNette."""
    backbone.eval()
    P = np.concatenate([torch.softmax(orig_fc(backbone(X[i:i + bs].to(device))), 1).cpu().numpy()
                        for i in range(0, len(X), bs)])
    aucs = [roc_auc_score((y == ci).astype(int), P[:, idx])
            for ci, idx in enumerate(IMAGENETTE_IMAGENET_IDX) if len(set((y == ci).tolist())) > 1]
    return float(np.mean(aucs))


def run_loo_once(freeze, pretrained, seed, tr_clean, te_clean, types, imnet, device, epochs, lr):
    """One full leave-one-corruption-out pass at a given seed. Returns folds + means."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    folds = {}
    for held in types:
        train_types = [t for t in types if t != held]
        imgs, labs, ts = [], [], 0
        for img in tr_clean:
            imgs.append(img); labs.append(0)
            for t in train_types:
                imgs.append(corrupt(img, t, float(rng.choice(BUDGETS)), rng, ts)); labs.append(1); ts += 1
        model = Net(freeze, pretrained)
        train(model, preprocess(imgs), np.array(labs), device, epochs, lr)

        te_imgs, te_lab = list(te_clean), [0] * len(te_clean)
        budget_idx = {b: [] for b in BUDGETS}
        for b in BUDGETS:
            for img in te_clean:
                budget_idx[b].append(len(te_imgs))
                te_imgs.append(corrupt(img, held, b, rng, ts)); te_lab.append(1); ts += 1
        s = score(model, preprocess(te_imgs), device)
        auroc = float(roc_auc_score(te_lab, s)) if len(set(te_lab)) > 1 else float("nan")
        rho = float(spearmanr(BUDGETS, [float(np.mean(s[budget_idx[b]])) for b in BUDGETS])[0])
        rec = {"rho": rho, "auroc": auroc}
        if imnet is not None:
            rec["imagenet_auc"] = imagenet_competence(
                model.backbone, imnet["orig_fc"], imnet["X"], imnet["y"], device)
        folds[held] = rec
    rs = [f["rho"] for f in folds.values() if f["rho"] == f["rho"]]
    au = [f["auroc"] for f in folds.values() if f["auroc"] == f["auroc"]]
    mimg = (float(np.mean([f["imagenet_auc"] for f in folds.values()]))
            if imnet is not None else None)
    return folds, float(np.mean(rs)), float(np.mean(au)), mimg


def ms(v):  # mean,std helper
    a = np.array(v, float)
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-root", default="/tmp/cdata")
    ap.add_argument("--types", nargs="+", default=CORRUPTIONS)
    ap.add_argument("--treatments", nargs="+", default=list(TREATMENTS))
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="42,123,456", help="comma-separated training seeds")
    ap.add_argument("--imagenet-dir", default=None,
                    help="ImageNette val dir (root/<wnid>/*.JPEG); enables feature-drift measure")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in str(args.seeds).split(",")]

    clean = load_clean(args.data_root, args.n_train + args.n_test)
    tr_clean, te_clean = clean[:args.n_train], clean[args.n_train:args.n_train + args.n_test]
    print(f"clean images: {len(tr_clean)} train / {len(te_clean)} test; seeds={seeds}", flush=True)

    imnet = None
    if args.imagenet_dir:
        orig = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        orig_fc = orig.fc.to(args.device).eval(); orig.fc = nn.Identity()
        im_imgs, im_y = load_imagenette(args.imagenet_dir)
        im_X = preprocess(im_imgs)
        base = imagenet_competence(orig.to(args.device), orig_fc, im_X, im_y, args.device)
        imnet = {"orig_fc": orig_fc, "X": im_X, "y": im_y, "baseline_auc": base}
        print(f"ImageNet competence baseline (pretrained backbone): AUC={base:.3f} "
              f"over {len(im_y)} ImageNette imgs", flush=True)

    results = {}
    for treat in args.treatments:
        freeze, pretrained = TREATMENTS[treat]
        per_seed = []
        for sd in seeds:
            folds, mrho, mau, mimg = run_loo_once(
                freeze, pretrained, sd, tr_clean, te_clean, args.types, imnet,
                args.device, args.epochs, args.lr)
            per_seed.append({"seed": sd, "mean_rho": mrho, "mean_auroc": mau,
                             "mean_imagenet_auc": mimg, "per_fold": folds})
            print(f"[{treat} seed={sd}] rho={mrho:+.3f} auroc={mau:.3f}"
                  + (f" imagenet_auc={mimg:.3f}" if imnet else ""), flush=True)
        rho_m, rho_s = ms([p["mean_rho"] for p in per_seed])
        au_m, au_s = ms([p["mean_auroc"] for p in per_seed])
        entry = {"per_seed": per_seed, "auroc_mean": au_m, "auroc_std": au_s,
                 "rho_mean": rho_m, "rho_std": rho_s, "seeds": seeds}
        if imnet is not None:
            entry["imagenet_auc_mean"], entry["imagenet_auc_std"] = ms(
                [p["mean_imagenet_auc"] for p in per_seed])
        results[treat] = entry
        print(f"=== {treat}: AUROC={au_m:.3f}+/-{au_s:.3f}  rho={rho_m:+.3f}+/-{rho_s:.3f}"
              + (f"  imagenet_AUC={entry['imagenet_auc_mean']:.3f}+/-{entry['imagenet_auc_std']:.3f}"
                 if imnet else "") + " ===", flush=True)

    if imnet is not None:
        results["imagenet_baseline_auc"] = imnet["baseline_auc"]
    args.out.write_text(json.dumps(results, indent=2))

    print(f"\nwrote {args.out}\nCAUSAL COMPARISON (LOO, mean +/- std over {len(seeds)} seeds):")
    for t in args.treatments:
        r = results[t]
        line = (f"  {t:9s} corrupt_AUROC={r['auroc_mean']:.3f}+/-{r['auroc_std']:.3f}  "
                f"corrupt_rho={r['rho_mean']:+.3f}+/-{r['rho_std']:.3f}")
        if imnet is not None:
            line += (f"  imagenet_AUC={r['imagenet_auc_mean']:.3f}+/-{r['imagenet_auc_std']:.3f}"
                     f" (baseline {imnet['baseline_auc']:.3f})")
        print(line)


if __name__ == "__main__":
    main()
