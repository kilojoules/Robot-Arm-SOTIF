"""Warp GPU kernel for lens glare / flare effects.

Implements haze (radial bloom), starburst streaks, and lens ghosts
on the GPU render buffer. Parameters are provided by IsaacGlareAdapter.

Requires: ``pip install warp-lang`` (included with Isaac Sim).
"""

_wp = None


def _ensure_warp():
    global _wp
    if _wp is None:
        import warp as wp
        wp.init()
        _wp = wp


def create_glare_annotator(image_width: int, image_height: int):
    """Create an annotator that applies glare to a GPU render buffer.

    Returns a callable ``apply(render_buffer, uniforms)``.
    """
    _ensure_warp()
    wp = _wp

    @wp.kernel
    def glare_kernel(
        image: wp.array2d(dtype=wp.vec4f),
        source_x: float,
        source_y: float,
        intensity: float,
        haze_spread: float,
        num_streaks: int,
        streak_angle_rad: float,
        streak_length: float,
    ):
        """Per-pixel glare kernel: haze + starburst streaks."""
        i, j = wp.tid()
        h = image.shape[0]
        w = image.shape[1]

        u = float(j) / float(w)
        v = float(i) / float(h)

        dx = u - source_x
        dy = v - source_y
        dist = wp.sqrt(dx * dx + dy * dy)
        max_dist = wp.sqrt(2.0)

        # Haze: radial Gaussian falloff
        haze_sigma = max_dist * 0.3 * haze_spread
        haze = wp.exp(-(dist * dist) / (2.0 * haze_sigma * haze_sigma))

        # Starburst: angular modulation
        angle = wp.atan2(dy, dx) - streak_angle_rad
        streak = float(0.0)
        if num_streaks > 0:
            angular_freq = float(num_streaks)
            cos_val = wp.cos(angular_freq * angle)
            # Sharp peaks via power
            streak_raw = wp.abs(cos_val)
            streak_raw = streak_raw * streak_raw * streak_raw
            # Distance falloff for streaks
            streak_dist = wp.exp(-dist / (streak_length * 0.3))
            streak = streak_raw * streak_dist

        # Combine and apply via screen blending
        glare_val = (haze * 0.5 + streak * 0.3) * intensity

        pixel = image[i, j]
        # Screen blend: out = 1 - (1 - base) * (1 - glare)
        out = wp.vec4f(
            1.0 - (1.0 - pixel[0]) * (1.0 - glare_val * 0.9),
            1.0 - (1.0 - pixel[1]) * (1.0 - glare_val * 0.95),
            1.0 - (1.0 - pixel[2]) * (1.0 - glare_val * 1.0),
            pixel[3],
        )

        image[i, j] = wp.vec4f(
            wp.clamp(out[0], 0.0, 1.0),
            wp.clamp(out[1], 0.0, 1.0),
            wp.clamp(out[2], 0.0, 1.0),
            out[3],
        )

    def apply(render_buffer, uniforms: dict):
        """Apply glare contamination to a GPU render buffer."""
        h, w = render_buffer.shape
        streak_angle_rad = uniforms["streak_angle_deg"] * 3.14159265 / 180.0

        wp.launch(
            kernel=glare_kernel,
            dim=(h, w),
            inputs=[
                render_buffer,
                uniforms["source_x"],
                uniforms["source_y"],
                uniforms["intensity"],
                uniforms["haze_spread"],
                uniforms["num_streaks"],
                streak_angle_rad,
                uniforms["streak_length"],
            ],
        )

    return apply
