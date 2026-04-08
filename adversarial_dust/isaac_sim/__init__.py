"""Isaac Sim integration for photorealistic sim-to-real transfer validation.

Provides GPU-accelerated evaluation of manipulation policies under
physically-based lens contamination rendered via NVIDIA Omniverse RTX.

Three capabilities beyond the CPU post-processing pipeline:

1. **Physics-based lens contamination** — Warp kernels apply fingerprint/glare
   effects on the GPU render buffer before CPU readback, producing
   physically-correct light-transport interactions with scene geometry.

2. **Massively parallel evaluation** — Isaac Lab vectorized environments
   evaluate an entire CMA-ES population simultaneously on a single GPU,
   reducing envelope prediction wall-clock time by an order of magnitude.

3. **Sim-to-real calibration** — differentiable parameter fitting maps
   simulated contamination to real-world lens photographs, constraining
   adversarial search to physically-realizable patterns and enabling
   certified safety guarantees on real hardware.
"""
