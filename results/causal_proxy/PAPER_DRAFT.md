# Draft prose for the frozen-vs-fine-tuned control (for editing into the full paper)

Academic voice, ready to adapt. Numbers from `results/causal_proxy/` (gbar jobs
28794481 / 28795265). Treat as a starting point, not final text.

---

## Methods (disentangling architecture, pretraining, and feature adaptation)

To determine *why* the frozen-backbone monitor generalizes across corruption
types — the visual representation supplied by ImageNet pretraining, the
ResNet-18 architecture itself, or the act of freezing — we run a controlled
comparison in which only the backbone-training treatment varies. Because the
high-fidelity SAPIEN renderer used for the robot study is incompatible with
current GPU drivers, we conduct this control in a decoupled, simulator-free
setting that isolates the representation question: we apply the same nine
corruption families used throughout this work to natural images (STL-10) and
train the identical ResNet-18 + lightweight head to detect corruption, under the
same leave-one-corruption-out (LOO) protocol and the same severity-ranking
(Spearman \rho) and discrimination (AUROC) metrics. Three treatments are
compared under identical data, folds, schedule, and seeds: (i) *frozen* — the
ImageNet-pretrained backbone is held fixed and only the head is trained;
(ii) *fine-tuned* — the same pretrained backbone is trained end-to-end; and
(iii) *from-scratch* — the same architecture trained from random initialization.
Each treatment is evaluated over three seeds and four training-set sizes
(n = 30, 100, 300, 1000 base images) to characterize the data regime.

We additionally quantify how far each treatment *moves* the representation with a
feature-drift probe: after training, the original 1000-way ImageNet
classification head is re-attached to the trained backbone and evaluated on held-
out ImageNet classes (ImageNette, macro one-vs-rest AUC). A backbone that retains
its general visual representation scores near the pretrained baseline; one that
has specialized away from it scores lower. For the frozen treatment this quantity
is fixed at the baseline by construction and serves as a sanity check; for the
other treatments it measures catastrophic forgetting of general features.

## Results

Pretraining is necessary for cross-corruption transfer: the from-scratch model is
worst at every training-set size (held-out AUROC 0.60–0.69), and its re-attached
ImageNet AUC is ≈ 0.50 (chance), confirming its backbone never learns
transferable features at these data scales. More informatively, the relationship
between freezing and fine-tuning is governed by the data regime (Fig. X, left).
The fine-tuned model's advantage over the frozen model grows monotonically with
data — the held-out-AUROC difference (fine-tuned minus frozen) is −0.011, +0.038,
+0.078, and +0.157 at n = 30, 100, 300, and 1000 — so fine-tuning only overtakes
freezing given sufficient data, with a crossover near n ≈ 50–100; in the scarcest
regime the frozen model is at least as accurate. The frozen model's accuracy is
essentially flat across data sizes (0.71–0.74), consistent with its small
(≈ 33k-parameter) trainable capacity.

The feature-drift probe explains the cost of that fine-tuning advantage (Fig. X,
right). The frozen backbone retains its full general representation (ImageNet AUC
0.997, baseline 0.998) at every data size, whereas the fine-tuned backbone's
retained ImageNet AUC falls steadily as data grows (0.975, 0.765, 0.624, 0.538 at
n = 30, 100, 300, 1000). Fine-tuning thus purchases higher in-distribution
corruption discrimination by progressively overwriting the general visual
features — the very features the monitor must rely on for corruptions it has
never encountered.

## Discussion

These controls refine the paper's central claim. They do not support the strong
reading that frozen features are uniquely responsible for generalization: given
ample data, fine-tuning the same backbone detects corruptions more accurately.
What they do establish is sharper and directly relevant to runtime safety
monitoring: pretrained features are essential, and in the data-limited regime
that characterizes safety-monitor training, freezing them matches or exceeds
fine-tuning while fully preserving the general representation that fine-tuning
erodes. This reconciles the two settings — the robot monitor, trained on limited
data, succeeds with a frozen backbone (mean \rho = 0.79), exactly where the data-
regime curve predicts freezing to be the preferred choice. For a monitor whose
purpose is to flag *unseen* hazards, preserving general visual features is not
merely competitive but desirable, since specialization to the training
corruptions is precisely the failure mode a safety monitor must avoid.

(Caveats to keep: the proxy task is corruption *detection* on natural images, not
robot-failure prediction; the n = 30 frozen advantage is within one standard
deviation, so the robust statement is the monotone trend rather than a win at the
smallest size; severity-ranking \rho over five budget points per fold is noisy and
AUROC is the reliable metric.)
