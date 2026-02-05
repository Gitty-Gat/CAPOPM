"""
Live visualization for the flat prior simulator.
"""

from __future__ import annotations

import logging
from typing import List, Sequence


import matplotlib

# Use Agg by default for headless environments; switch to default if live.

from matplotlib import animation  # noqa: E402
import numpy as np  # noqa: E402


def make_live_animation(
    mid_prices: Sequence[float],
    regimes: Sequence[int],
    posterior_means: Sequence[float],
    weights: Sequence[float],
    save_path: str,
    fps: int = 30,
    live: bool = False,
    logger: logging.Logger | None = None,
):
    import matplotlib
    if live:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import animation


    log = logger or logging.getLogger(__name__)
    n = len(mid_prices)
    x = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_price, ax_post, ax_weight = axes

    price_line, = ax_price.plot([], [], lw=1.5, color="steelblue")
    regime_shading = ax_price.fill_between([], [], color="lightgray", alpha=0.3)
    ax_price.set_ylabel("Mid Price")

    post_line, = ax_post.plot([], [], lw=1.5, color="darkorange")
    ax_post.axhline(0.25, ls="--", color="gray", alpha=0.5)
    ax_post.axhline(0.5, ls="-.", color="gray", alpha=0.5)
    ax_post.axhline(0.75, ls="--", color="gray", alpha=0.5)
    ax_post.set_ylabel("Posterior mean p_t")
    ax_post.set_ylim(0, 1)

    weight_line, = ax_weight.plot([], [], lw=1.5, color="seagreen")
    ax_weight.set_ylabel("Mixture weight w_t")
    ax_weight.set_ylim(0, 1)
    ax_weight.set_xlabel("Event index")

    def init():
        price_line.set_data([], [])
        post_line.set_data([], [])
        weight_line.set_data([], [])
        return price_line, post_line, weight_line

    def animate(i):
        idx = i + 1
        price_line.set_data(x[:idx], mid_prices[:idx])
        post_line.set_data(x[:idx], posterior_means[:idx])
        weight_line.set_data(x[:idx], weights[:idx])

        # regime shading: quick approach using background color toggle
        # Remove any collection artists (e.g., regime shading) in a version-safe way
        for coll in list(ax_price.collections):
            try:
                coll.remove()
            except Exception:
                pass

        if regimes:
            regime_arr = np.array(regimes[:idx])
            unique = np.unique(regime_arr)
            for r in unique:
                mask = regime_arr == r
                if mask.any():
                    start_end = _contiguous_segments(np.where(mask)[0])
                    for s, e in start_end:
                        ax_price.axvspan(s, e, color="lightgray", alpha=0.15 + 0.1 * (r % 3))
        return price_line, post_line, weight_line

    frames = n
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=1000 / fps, blit=False)
    try:
        anim.save(save_path, fps=fps, dpi=120)
    except Exception as e:
        log.exception("Animation save failed; continuing without video. Error: %s", e)


    log.info("Saved live animation to %s", save_path)

    if live:
        plt.show()         # blocking, keeps window alive
    else:
        plt.close(fig)


def _contiguous_segments(indices: np.ndarray) -> List[tuple[int, int]]:
    """Convert sorted indices into contiguous (start, end) segments."""
    if len(indices) == 0:
        return []
    segments = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev))
        start = idx
        prev = idx
    segments.append((start, prev))
    return segments

