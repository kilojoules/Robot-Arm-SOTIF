"""Feature-space similarity analysis between corruption types.

Computes ResNet-18 feature embeddings for each corruption type, builds
a pairwise cosine similarity matrix, and predicts which corruptions will
be easiest/hardest to generalize to in the LOO analysis.

Can run in two modes:
  1. Synthetic: Apply corruptions to a sample image (no sim needed)
  2. From data: Use collected LOO .npz files

Usage:
    # Synthetic mode (works anywhere with torch/torchvision):
    python -m adversarial_dust.feature_similarity \
        --mode synthetic --sample-image path/to/image.png

    # From collected LOO data:
    python -m adversarial_dust.feature_similarity \
        --mode data --data-dir results/safety_predictor/loo_data

    # Predict LOO difficulty (run BEFORE the LOO experiment):
    python -m adversarial_dust.feature_similarity \
        --mode synthetic --predict-loo
"""

import argparse
import json
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np


# All 9 corruption types in the LOO study
ALL_TYPES = [
    "fingerprint", "glare", "rain", "gaussian_noise", "jpeg",
    "motion_blur", "defocus_blur", "dust_camera", "low_light",
]

CATEGORIES = {
    "fingerprint": "Lens contact",
    "rain": "Lens contact",
    "glare": "Optical",
    "defocus_blur": "Optical",
    "motion_blur": "Blur",
    "gaussian_noise": "Sensor",
    "dust_camera": "Environmental",
    "low_light": "Illumination",
    "jpeg": "Digital",
}


def extract_backbone_features(images: list, device: str = "cpu") -> np.ndarray:
    """Pass images through frozen ResNet-18 backbone, return feature vectors.

    Args:
        images: List of HWC uint8 numpy arrays.
        device: 'cpu' or 'cuda'.

    Returns:
        (N, 512) array of feature vectors.
    """
    import torch
    from torchvision import models, transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load frozen ResNet-18 backbone (everything before fc)
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    resnet.to(device)

    features = []
    with torch.no_grad():
        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)
            feat = resnet(tensor).cpu().numpy().flatten()
            features.append(feat)

    return np.array(features)


def generate_corrupted_images(
    base_image: np.ndarray,
    corruption_types: list,
    budget_levels: list,
    n_samples: int = 5,
) -> dict:
    """Generate corrupted versions of a base image for each corruption type.

    Returns dict mapping corruption_type -> list of corrupted images.
    """
    from adversarial_dust.envelope_predictor import make_occlusion_model
    from adversarial_dust.config import EnvelopeExperimentConfig

    config = EnvelopeExperimentConfig()
    image_shape = base_image.shape

    corrupted = {}
    rng = np.random.default_rng(42)

    for ctype in corruption_types:
        images = []
        for budget in budget_levels:
            model = make_occlusion_model(ctype, config, image_shape, budget)
            for _ in range(n_samples):
                params = model.get_random_params(rng)
                img = model.apply(base_image.copy(), params, timestep=0)
                images.append(img)
        corrupted[ctype] = images

    return corrupted


def compute_similarity_matrix(
    feature_centroids: dict,
    corruption_types: list,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    n = len(corruption_types)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vi = feature_centroids[corruption_types[i]]
            vj = feature_centroids[corruption_types[j]]
            cos_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8)
            matrix[i, j] = cos_sim

    return matrix


def predict_loo_difficulty(
    feature_centroids: dict,
    corruption_types: list,
) -> list:
    """Predict LOO difficulty: corruptions with LOW mean similarity to others
    should be HARDEST to generalize to (lowest expected rho).

    Returns list of (corruption, mean_similarity_to_others) sorted by
    predicted difficulty (hardest first).
    """
    predictions = []
    for held_out in corruption_types:
        others = [t for t in corruption_types if t != held_out]
        v_held = feature_centroids[held_out]
        sims = []
        for other in others:
            v_other = feature_centroids[other]
            cos_sim = np.dot(v_held, v_other) / (
                np.linalg.norm(v_held) * np.linalg.norm(v_other) + 1e-8)
            sims.append(cos_sim)
        mean_sim = float(np.mean(sims))
        predictions.append({
            "corruption": held_out,
            "category": CATEGORIES.get(held_out, "Unknown"),
            "mean_similarity_to_training": mean_sim,
            "min_similarity": float(np.min(sims)),
            "max_similarity": float(np.max(sims)),
        })

    # Sort by mean similarity (lowest = hardest to generalize to)
    predictions.sort(key=lambda x: x["mean_similarity_to_training"])
    return predictions


def run_synthetic(base_image: np.ndarray, output_dir: Path,
                  device: str = "cpu"):
    """Run the full analysis in synthetic mode."""
    budget_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    corruption_types = ALL_TYPES

    print("Generating corrupted images...", flush=True)
    corrupted = generate_corrupted_images(
        base_image, corruption_types, budget_levels, n_samples=3)

    # Also generate clean features
    print("Extracting clean features...", flush=True)
    clean_features = extract_backbone_features([base_image] * 5, device=device)
    clean_centroid = clean_features.mean(axis=0)

    # Extract features per corruption type
    print("Extracting corruption features...", flush=True)
    raw_centroids = {}
    diff_centroids = {}

    for ctype in corruption_types:
        images = corrupted[ctype]
        features = extract_backbone_features(images, device=device)
        centroid = features.mean(axis=0)
        raw_centroids[ctype] = centroid
        # Mintun-style: feature difference = f(corrupted) - f(clean)
        diff_centroids[ctype] = centroid - clean_centroid
        print(f"  {ctype}: {len(images)} images, "
              f"||feat||={np.linalg.norm(centroid):.1f}, "
              f"||diff||={np.linalg.norm(centroid - clean_centroid):.1f}",
              flush=True)

    # Compute similarity matrices
    print("\nComputing similarity matrices...", flush=True)
    raw_matrix = compute_similarity_matrix(raw_centroids, corruption_types)
    diff_matrix = compute_similarity_matrix(diff_centroids, corruption_types)

    # Predict LOO difficulty
    print("\n" + "=" * 60, flush=True)
    print("LOO DIFFICULTY PREDICTIONS (feature-space similarity)", flush=True)
    print("=" * 60, flush=True)

    print("\n--- Raw feature similarity ---", flush=True)
    raw_predictions = predict_loo_difficulty(raw_centroids, corruption_types)
    print(f"  {'Rank':>4} {'Corruption':>15} {'Category':>14} "
          f"{'Mean Sim':>9} {'Predicted':>12}", flush=True)
    print(f"  {'-' * 58}", flush=True)
    for rank, pred in enumerate(raw_predictions):
        difficulty = "HARDEST" if rank < 3 else ("EASIEST" if rank >= 6 else "MODERATE")
        print(f"  {rank+1:>4} {pred['corruption']:>15} "
              f"{pred['category']:>14} "
              f"{pred['mean_similarity_to_training']:>9.4f} "
              f"{difficulty:>12}", flush=True)

    print("\n--- Feature-difference similarity (Mintun-style) ---", flush=True)
    diff_predictions = predict_loo_difficulty(diff_centroids, corruption_types)
    print(f"  {'Rank':>4} {'Corruption':>15} {'Category':>14} "
          f"{'Mean Sim':>9} {'Predicted':>12}", flush=True)
    print(f"  {'-' * 58}", flush=True)
    for rank, pred in enumerate(diff_predictions):
        difficulty = "HARDEST" if rank < 3 else ("EASIEST" if rank >= 6 else "MODERATE")
        print(f"  {rank+1:>4} {pred['corruption']:>15} "
              f"{pred['category']:>14} "
              f"{pred['mean_similarity_to_training']:>9.4f} "
              f"{difficulty:>12}", flush=True)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "corruption_types": corruption_types,
        "categories": CATEGORIES,
        "raw_similarity_matrix": raw_matrix.tolist(),
        "diff_similarity_matrix": diff_matrix.tolist(),
        "raw_predictions": raw_predictions,
        "diff_predictions": diff_predictions,
        "raw_centroids_norm": {k: float(np.linalg.norm(v))
                               for k, v in raw_centroids.items()},
        "diff_centroids_norm": {k: float(np.linalg.norm(v))
                                for k, v in diff_centroids.items()},
    }

    results_path = output_dir / "feature_similarity.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}", flush=True)

    # Generate heatmap figure
    try:
        _plot_heatmaps(raw_matrix, diff_matrix, corruption_types, output_dir)
    except Exception as e:
        print(f"Could not generate plots: {e}", flush=True)

    return results


def run_from_data(data_dir: Path, output_dir: Path, device: str = "cpu"):
    """Run analysis from collected LOO .npz files."""
    corruption_types = []
    raw_centroids = {}

    # Load clean baseline if available
    clean_path = data_dir / "clean.npz"
    clean_centroid = None
    if clean_path.exists():
        d = np.load(str(clean_path))
        clean_features = extract_backbone_features(list(d["images"]), device)
        clean_centroid = clean_features.mean(axis=0)

    for npz_path in sorted(data_dir.glob("*.npz")):
        ctype = npz_path.stem
        if ctype == "clean" or ctype == "training_data":
            continue
        if ctype not in ALL_TYPES:
            continue

        print(f"Processing {ctype}...", flush=True)
        d = np.load(str(npz_path))
        images = list(d["images"])

        # Subsample if large
        if len(images) > 100:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(images), 100, replace=False)
            images = [images[i] for i in idx]

        features = extract_backbone_features(images, device)
        centroid = features.mean(axis=0)
        raw_centroids[ctype] = centroid
        corruption_types.append(ctype)
        print(f"  {ctype}: {len(images)} images, "
              f"||feat||={np.linalg.norm(centroid):.1f}", flush=True)

    if not corruption_types:
        print("No data found!", flush=True)
        return

    # Compute feature differences if clean baseline available
    diff_centroids = {}
    if clean_centroid is not None:
        for ctype in corruption_types:
            diff_centroids[ctype] = raw_centroids[ctype] - clean_centroid

    # Same analysis as synthetic mode
    raw_matrix = compute_similarity_matrix(raw_centroids, corruption_types)

    print("\nLOO DIFFICULTY PREDICTIONS:", flush=True)
    predictions = predict_loo_difficulty(raw_centroids, corruption_types)
    for rank, pred in enumerate(predictions):
        difficulty = "HARDEST" if rank < 3 else ("EASIEST" if rank >= len(corruption_types) - 3 else "MODERATE")
        print(f"  {rank+1}. {pred['corruption']:>15} "
              f"(sim={pred['mean_similarity_to_training']:.4f}) "
              f"-> {difficulty}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "corruption_types": corruption_types,
        "raw_similarity_matrix": raw_matrix.tolist(),
        "predictions": predictions,
    }
    if diff_centroids:
        diff_matrix = compute_similarity_matrix(diff_centroids, corruption_types)
        diff_predictions = predict_loo_difficulty(diff_centroids, corruption_types)
        results["diff_similarity_matrix"] = diff_matrix.tolist()
        results["diff_predictions"] = diff_predictions

    with open(output_dir / "feature_similarity.json", "w") as f:
        json.dump(results, f, indent=2)

    try:
        diff_mat = np.array(results.get("diff_similarity_matrix", [])) \
            if "diff_similarity_matrix" in results else None
        _plot_heatmaps(raw_matrix, diff_mat, corruption_types, output_dir)
    except Exception as e:
        print(f"Could not generate plots: {e}", flush=True)


def _plot_heatmaps(raw_matrix, diff_matrix, corruption_types, output_dir):
    """Generate similarity heatmap figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [t.replace("_", "\n") for t in corruption_types]
    cats = [CATEGORIES.get(t, "?") for t in corruption_types]

    n_plots = 2 if diff_matrix is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]

    for ax, matrix, title in zip(
        axes,
        [raw_matrix] + ([diff_matrix] if diff_matrix is not None else []),
        ["Raw Feature Similarity", "Feature-Difference Similarity"],
    ):
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Annotate cells
        for i in range(len(corruption_types)):
            for j in range(len(corruption_types)):
                val = matrix[i, j]
                color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")

    plt.tight_layout()
    path = output_dir / "feature_similarity_heatmap.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Feature-space similarity analysis for corruption types")
    parser.add_argument("--mode", choices=["synthetic", "data"],
                        default="synthetic")
    parser.add_argument("--sample-image", type=str, default=None,
                        help="Path to a sample image for synthetic mode. "
                             "If not provided, generates a random image.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to LOO data directory (for data mode)")
    parser.add_argument("--output-dir", type=str,
                        default="results/feature_similarity")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.mode == "synthetic":
        if args.sample_image:
            base_image = cv2.imread(args.sample_image)
            if base_image is None:
                raise FileNotFoundError(f"Could not load {args.sample_image}")
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        else:
            # Generate a plausible robot-arm-like image (gradient + shapes)
            print("No sample image provided, generating synthetic base image",
                  flush=True)
            rng = np.random.default_rng(42)
            base_image = rng.integers(80, 200, (480, 640, 3), dtype=np.uint8)
            # Add some structure (horizontal gradient + rectangle)
            for y in range(480):
                base_image[y, :, :] = np.clip(
                    base_image[y, :, :].astype(int) + (y - 240) // 3, 0, 255)
            cv2.rectangle(base_image, (200, 150), (440, 350), (60, 120, 180), -1)
            cv2.circle(base_image, (320, 250), 40, (200, 80, 80), -1)

        run_synthetic(base_image, output_dir, device=args.device)

    elif args.mode == "data":
        if not args.data_dir:
            raise ValueError("--data-dir required for data mode")
        run_from_data(Path(args.data_dir), output_dir, device=args.device)


if __name__ == "__main__":
    main()
