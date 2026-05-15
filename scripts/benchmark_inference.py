"""Benchmark single-frame inference latency for the safety monitor.

Instantiates SafetyPredictorCNN with the paper's architecture (frozen
ResNet-18 + 33K head), runs a warmup, then times many forward passes on a
synthetic 224x224 uint8 image. Reports median, p95, and throughput.

Usage:
    python scripts/benchmark_inference.py                    # auto-detect device
    python scripts/benchmark_inference.py --device cuda      # force GPU
    python scripts/benchmark_inference.py --batch-size 8     # time a batch

Output:
    results/inference_benchmark/<device>.json
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def pick_device(explicit: str) -> str:
    if explicit != "auto":
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def gpu_name(device: str):
    try:
        import torch
        if device == "cuda":
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def benchmark(device: str, batch_size: int, n_warmup: int, n_iter: int,
              backbone: str):
    from adversarial_dust.safety_predictor import SafetyPredictorCNN
    import torch

    predictor = SafetyPredictorCNN(backbone=backbone).to(device)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    batch = [frame] * batch_size

    # Warmup (important on GPU)
    for _ in range(n_warmup):
        predictor.predict_batch(batch)
    if device == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_iter):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.predict_batch(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    p50 = statistics.median(times_ms)
    p95 = times_ms[int(0.95 * len(times_ms))]
    p99 = times_ms[int(0.99 * len(times_ms))]
    mean = sum(times_ms) / len(times_ms)
    throughput = batch_size * 1000.0 / mean

    return {
        "device": device,
        "gpu_name": gpu_name(device),
        "backbone": backbone,
        "batch_size": batch_size,
        "n_warmup": n_warmup,
        "n_iter": n_iter,
        "latency_ms": {
            "p50": p50, "p95": p95, "p99": p99, "mean": mean,
            "min": times_ms[0], "max": times_ms[-1],
        },
        "throughput_fps": throughput,
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet50", "vit_b_16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iter", type=int, default=200)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/inference_benchmark"))
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Benchmarking backbone={args.backbone} batch={args.batch_size} "
          f"device={device}")
    result = benchmark(device, args.batch_size, args.warmup, args.iter,
                       args.backbone)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.backbone}_{device}_bs{args.batch_size}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    lat = result["latency_ms"]
    print(f"  p50={lat['p50']:.2f} ms   p95={lat['p95']:.2f} ms   "
          f"p99={lat['p99']:.2f} ms")
    print(f"  throughput: {result['throughput_fps']:.1f} fps")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
