"""Warp GPU kernel for physics-based fingerprint contamination.

Implements the Gu et al. (SIGGRAPH Asia 2009) image formation model on the
GPU render buffer via NVIDIA Warp:

    I_out = I_in * (1 - attenuation) + grease_color * attenuation
            + scatter_tint * attenuation^2 * scene_brightness

The kernel receives uniform parameters from IsaacFingerprintAdapter and
operates on the GPU-resident image tensor before CPU readback.

Requires: ``pip install warp-lang`` (included with Isaac Sim).
"""

# All actual Warp code is guarded behind lazy import so that tests and
# planning tools can import this module without an NVIDIA GPU.

_wp = None


def _ensure_warp():
    global _wp
    if _wp is None:
        import warp as wp
        wp.init()
        _wp = wp


def create_fingerprint_annotator(image_width: int, image_height: int):
    """Create an omni.replicator annotator that applies fingerprint contamination.

    Returns a callable that takes (render_product, uniform_dict) and applies
    the contamination in-place on the GPU render buffer.

    This is registered once per camera; the uniform dict is updated each
    timestep by the evaluator.
    """
    _ensure_warp()
    wp = _wp

    @wp.kernel
    def fingerprint_kernel(
        image: wp.array2d(dtype=wp.vec4f),
        # Per-print arrays (flattened)
        centers_x: wp.array(dtype=float),
        centers_y: wp.array(dtype=float),
        scales: wp.array(dtype=float),
        opacities: wp.array(dtype=float),
        # Scalar uniforms
        num_prints: int,
        defocus_sigma: float,
        grease_r: float,
        grease_g: float,
        grease_b: float,
        scatter_r: float,
        scatter_g: float,
        scatter_b: float,
    ):
        """Per-pixel fingerprint contamination kernel."""
        i, j = wp.tid()
        h = image.shape[0]
        w = image.shape[1]

        # Normalized pixel coordinates
        u = float(j) / float(w)
        v = float(i) / float(h)

        # Compute total attenuation from all prints
        attn = float(0.0)
        for p in range(num_prints):
            dx = u - centers_x[p]
            dy = v - centers_y[p]
            dist_sq = dx * dx + dy * dy
            radius = scales[p]
            if dist_sq < radius * radius:
                # Smooth falloff
                falloff = 1.0 - wp.sqrt(dist_sq) / radius
                attn = wp.max(attn, opacities[p] * falloff * falloff)

        # Apply Gu et al. image formation model
        pixel = image[i, j]
        scene_brightness = (pixel[0] + pixel[1] + pixel[2]) / 3.0

        # Grease color contribution
        grease = wp.vec4f(
            grease_r * attn * 2.0,
            grease_g * attn * 2.0,
            grease_b * attn * 2.0,
            1.0,
        )

        # Scatter (veiling glare)
        scatter_strength = attn * attn * scene_brightness * 1.5
        scatter = wp.vec4f(
            scatter_r * scatter_strength,
            scatter_g * scatter_strength,
            scatter_b * scatter_strength,
            0.0,
        )

        # Composite
        out = wp.vec4f(
            pixel[0] * (1.0 - attn) + grease[0] + scatter[0],
            pixel[1] * (1.0 - attn) + grease[1] + scatter[1],
            pixel[2] * (1.0 - attn) + grease[2] + scatter[2],
            pixel[3],
        )

        # Clamp to [0, 1]
        image[i, j] = wp.vec4f(
            wp.clamp(out[0], 0.0, 1.0),
            wp.clamp(out[1], 0.0, 1.0),
            wp.clamp(out[2], 0.0, 1.0),
            out[3],
        )

    def apply(render_buffer, uniforms: dict):
        """Apply fingerprint contamination to a GPU render buffer.

        Args:
            render_buffer: wp.array2d of vec4f (RGBA float image on GPU).
            uniforms: Dict from IsaacFingerprintAdapter.translate().
        """
        h, w = render_buffer.shape
        n = uniforms["num_prints"]

        wp.launch(
            kernel=fingerprint_kernel,
            dim=(h, w),
            inputs=[
                render_buffer,
                wp.array(uniforms["centers"][:, 0], dtype=float),
                wp.array(uniforms["centers"][:, 1], dtype=float),
                wp.array(uniforms["scales"], dtype=float),
                wp.array(uniforms["opacities"], dtype=float),
                n,
                uniforms["defocus_sigma"],
                uniforms["grease_color"][0],
                uniforms["grease_color"][1],
                uniforms["grease_color"][2],
                uniforms["scatter_tint"][0],
                uniforms["scatter_tint"][1],
                uniforms["scatter_tint"][2],
            ],
        )

    return apply
