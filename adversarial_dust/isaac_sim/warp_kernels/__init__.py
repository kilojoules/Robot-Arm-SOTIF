"""Warp GPU kernels for lens contamination effects.

These kernels operate on Isaac Sim's GPU-resident render buffer
(via omni.replicator annotators) before the image is copied to CPU,
achieving physically-based contamination at render-pipeline speed.
"""
