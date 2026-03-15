#!/usr/bin/env python3
"""Generate figures for README: animated GIF comparing v3 (from-scratch) vs v4 (ResNet-18)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "safety_predictor"
OUT = ROOT / "docs" / "figures"


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_curve(results, occlusion_type):
    budgets, actual_fr, pred_pf = [], [], []
    for b_str in sorted(results[occlusion_type], key=float):
        r = results[occlusion_type][b_str]
        budgets.append(float(b_str) * 100)
        actual_fr.append((1 - r["actual_success_rate"]) * 100)
        pred_pf.append(r["mean_predicted_p_failure"] * 100)
    return np.array(budgets), np.array(actual_fr), np.array(pred_pf)


def make_animated_comparison(v3, v4):
    """Build animated GIF: frames progressively reveal v3 then v4 rain results."""
    budgets_v3, fr_v3, pf_v3 = extract_curve(v3, "rain")
    budgets_v4, fr_v4, pf_v4 = extract_curve(v4, "rain")

    # Color palette
    C_ACTUAL = "#1a1a2e"
    C_V3 = "#e94560"
    C_V4 = "#0f3460"
    C_BG = "#f8f9fa"
    C_GRID = "#dee2e6"

    frames = []

    def make_frame(n_v3=0, n_v4=0, show_rho_v3=False, show_rho_v4=False,
                   show_arrow=False, hold=False):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_BG)

        # Grid
        ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        # Actual failure rate (always shown)
        ax.plot(budgets_v4, fr_v4, "o-", color=C_ACTUAL, linewidth=2.5,
                markersize=8, label="Actual failure rate", zorder=5)

        # v3 predictions (progressive reveal)
        if n_v3 > 0:
            ax.plot(budgets_v3[:n_v3], pf_v3[:n_v3], "s--", color=C_V3,
                    linewidth=2, markersize=7, alpha=0.85,
                    label="CNN from scratch (v3)", zorder=4)

        # v4 predictions (progressive reveal)
        if n_v4 > 0:
            ax.plot(budgets_v4[:n_v4], pf_v4[:n_v4], "D-", color=C_V4,
                    linewidth=2.5, markersize=8,
                    label="ResNet-18 pretrained (v4)", zorder=6)

        # Spearman rho annotations
        if show_rho_v3:
            ax.annotate(
                r"$\rho = -0.62$" + "\n(anti-correlated!)",
                xy=(90, pf_v3[-1]), xytext=(60, 6),
                fontsize=11, fontweight="bold", color=C_V3,
                arrowprops=dict(arrowstyle="->", color=C_V3, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_V3, alpha=0.9),
                zorder=10,
            )

        if show_rho_v4:
            ax.annotate(
                r"$\rho = 0.90$  $p < 0.05$" + "\n(strong generalization)",
                xy=(90, pf_v4[-1]), xytext=(40, 26),
                fontsize=11, fontweight="bold", color=C_V4,
                arrowprops=dict(arrowstyle="->", color=C_V4, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_V4, alpha=0.9),
                zorder=10,
            )

        # Arrow showing improvement
        if show_arrow:
            ax.annotate(
                "", xy=(95, pf_v4[-1]), xytext=(95, pf_v3[-1]),
                arrowprops=dict(arrowstyle="<->", color="#27ae60",
                                lw=2.5, mutation_scale=15),
                zorder=10,
            )
            mid_y = (pf_v3[-1] + pf_v4[-1]) / 2
            ax.text(97, mid_y, "190x", fontsize=10, fontweight="bold",
                    color="#27ae60", ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#27ae60", alpha=0.9),
                    zorder=10)

        ax.set_xlabel("Occlusion Budget (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Failure / Predicted Failure (%)", fontsize=12,
                       fontweight="bold")
        ax.set_title("Cross-Corruption Generalization: Rain (Never Seen in Training)",
                      fontsize=13, fontweight="bold", pad=12)

        ax.set_xlim(5, 100)
        ax.set_ylim(-2, 85)
        ax.set_xticks([10, 30, 50, 70, 90])

        # Legend
        if n_v3 > 0 or n_v4 > 0:
            ax.legend(loc="upper left", fontsize=10, framealpha=0.95,
                      edgecolor=C_GRID)

        plt.tight_layout()

        # Render to PIL Image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = Image.frombytes("RGBA", (w, h),
                              fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)
        return img

    # Phase 0: Just the actual failure rate (8 frames = ~0.7s)
    for _ in range(8):
        frames.append(make_frame(n_v3=0, n_v4=0))

    # Phase 1: Draw v3 line point by point (5 frames, each ~0.4s)
    for i in range(1, 6):
        for _ in range(4):
            frames.append(make_frame(n_v3=i, n_v4=0))

    # Phase 2: Show v3 rho annotation, hold (12 frames = ~1s)
    for _ in range(12):
        frames.append(make_frame(n_v3=5, n_v4=0, show_rho_v3=True))

    # Phase 3: Draw v4 line point by point (5 frames)
    for i in range(1, 6):
        for _ in range(4):
            frames.append(make_frame(n_v3=5, n_v4=i, show_rho_v3=True))

    # Phase 4: Show v4 rho + arrow, hold (20 frames = ~1.7s)
    for _ in range(20):
        frames.append(make_frame(n_v3=5, n_v4=5, show_rho_v3=True,
                                  show_rho_v4=True, show_arrow=True))

    # Phase 5: Final hold (15 frames, then loop)
    for _ in range(15):
        frames.append(make_frame(n_v3=5, n_v4=5, show_rho_v3=True,
                                  show_rho_v4=True, show_arrow=True))

    return frames


def make_static_summary(v4):
    """Static figure: fingerprint (train) + rain (held-out) side by side."""
    C_ACTUAL = "#1a1a2e"
    C_FP = "#e94560"
    C_RAIN = "#0f3460"
    C_BG = "#f8f9fa"
    C_GRID = "#dee2e6"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    fig.patch.set_facecolor(C_BG)

    for ax, occ_type, color, title, rho, pval in [
        (ax1, "fingerprint", C_FP,
         "Fingerprint (Training Distribution)", 0.975, 0.005),
        (ax2, "rain", C_RAIN,
         "Rain (Held-Out, Never Seen)", 0.900, 0.037),
    ]:
        ax.set_facecolor(C_BG)
        ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        budgets, fr, pf = extract_curve(v4, occ_type)

        ax.bar(budgets - 3, fr, width=6, color=C_ACTUAL, alpha=0.7,
               label="Actual failure rate", zorder=3)
        ax.plot(budgets, pf, "D-", color=color, linewidth=2.5,
                markersize=9, label="Predicted P(fail)", zorder=5)

        ax.set_xlabel("Occlusion Budget (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Failure / Prediction (%)", fontsize=11,
                       fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(-2, max(max(fr), max(pf)) * 1.3 + 5)
        ax.set_xticks([10, 30, 50, 70, 90])

        ax.annotate(
            f"Spearman $\\rho$ = {rho:.3f}\n$p$ = {pval:.3f}",
            xy=(0.98, 0.95), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color=color,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color, alpha=0.95),
        )

        ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    plt.tight_layout(w_pad=3)
    return fig


def make_architecture_diagram():
    """Minimal architecture diagram."""
    C_BG = "#f8f9fa"
    fig, ax = plt.subplots(figsize=(12, 3), dpi=120)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.axis("off")

    boxes = [
        (0.08, "Camera\nImage\n(512x640)", "#adb5bd", "#495057"),
        (0.28, "ResNet-18\nBackbone\n(frozen, 11M)", "#74b9ff", "#0f3460"),
        (0.48, "FC 512->64\nReLU\nDropout 0.3", "#a29bfe", "#6c5ce7"),
        (0.65, "FC 64->1\nSigmoid", "#a29bfe", "#6c5ce7"),
        (0.82, "P(failure)\n[0, 1]", "#ff7675", "#d63031"),
    ]

    for x, text, bg, edge in boxes:
        ax.add_patch(plt.Rectangle(
            (x, 0.15), 0.14, 0.7, transform=ax.transAxes,
            facecolor=bg, edgecolor=edge, linewidth=2,
            clip_on=False, zorder=3, alpha=0.85,
        ))
        ax.text(x + 0.07, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=edge, zorder=4)

    # Arrows between boxes
    for x1, x2 in [(0.22, 0.28), (0.42, 0.48), (0.62, 0.65), (0.79, 0.82)]:
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="#495057",
                                     lw=2, mutation_scale=20),
                     zorder=5)

    ax.set_title("SafetyNet Architecture: Only 33K trainable parameters",
                  fontsize=13, fontweight="bold", pad=15, color="#1a1a2e")

    plt.tight_layout()
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    v3 = load_results(RESULTS / "evaluation_results_v3.json")
    v4 = load_results(RESULTS / "evaluation_results_v4.json")

    # 1. Animated comparison GIF
    print("Generating animated comparison GIF...", flush=True)
    gif_frames = make_animated_comparison(v3, v4)
    gif_path = OUT / "generalization_animation.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=85,  # ms per frame
        loop=0,
    )
    print(f"  -> {gif_path} ({len(gif_frames)} frames)")

    # 2. Static summary figure
    print("Generating static summary...", flush=True)
    fig = make_static_summary(v4)
    fig.savefig(OUT / "evaluation_summary.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {OUT / 'evaluation_summary.png'}")

    # 3. Architecture diagram
    print("Generating architecture diagram...", flush=True)
    fig = make_architecture_diagram()
    fig.savefig(OUT / "architecture.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {OUT / 'architecture.png'}")

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
