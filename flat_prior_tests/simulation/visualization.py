"""
Live visualization for the flat prior simulator.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np  # noqa: E402


def make_dual_panel_animation(
    ts: Sequence[float],
    mid_prices: Sequence[float],
    alpha_seq: Sequence[float],
    beta_seq: Sequence[float],
    capopm_price: Sequence[float],
    cutoff_index: int,
    save_path: str,
    fps: int = 30,
    max_frames: int = 900,
    grid_points: int = 120,
    rolling_window: int = 80,
    camera_elev: float = 25.0,
    camera_azim: float = -60.0,
    logger: logging.Logger | None = None,
):
    """
    Dual-panel MP4: top=3D posterior surface (rolling window), bottom=price overlay.
    Uses corrected posterior parameters for CAPOPM fair value overlay.
    """

    import math
    import os
    import time

    import imageio_ffmpeg
    import matplotlib
    from matplotlib import animation
    from matplotlib.animation import FFMpegWriter
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    log = logger or logging.getLogger(__name__)
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["animation.writer"] = "ffmpeg"

    n = min(len(mid_prices), len(alpha_seq), len(beta_seq), len(capopm_price), len(ts))
    if n == 0:
        log.warning("No data for dual-panel animation; skipping save to %s", save_path)
        return

    max_frames = max(10, min(max_frames, 900))
    stride = max(1, int(math.ceil(n / max_frames)))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    ts_ds = np.asarray(ts)[idx]
    ts_ds = ts_ds - ts_ds[0]
    mid_ds = np.asarray(mid_prices)[idx]
    cap_ds = np.asarray(capopm_price)[idx]
    alpha_ds = np.asarray(alpha_seq)[idx]
    beta_ds = np.asarray(beta_seq)[idx]

    cutoff_index = max(0, min(n, cutoff_index))
    cutoff_ds_idx = np.searchsorted(idx, cutoff_index, side="left")
    cutoff_time = ts_ds[min(cutoff_ds_idx, len(ts_ds) - 1)]

    p_grid = np.linspace(0.001, 0.999, grid_points)
    log_pdf = (np.outer(alpha_ds - 1.0, np.log(p_grid)) + np.outer(beta_ds - 1.0, np.log(1.0 - p_grid)))
    norm = np.log(np.exp(log_pdf).sum(axis=1, keepdims=True) + 1e-15)
    pdf_matrix = np.exp(log_pdf - norm)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])

    mid_line, = ax2d.plot([], [], color="steelblue", lw=1.5, label="Simulated mid")
    cap_line, = ax2d.plot([], [], color="darkorange", lw=1.5, label="CAPOPM fair value")
    ax2d.axvline(cutoff_time, color="gray", ls="--", alpha=0.5, lw=1)
    ax2d.set_ylabel("Price ($)")
    ax2d.set_xlabel("Time (days, normalized)")
    ax2d.legend(loc="upper left")

    surface = None

    def _update_surface(frame_idx: int):
        nonlocal surface
        start = max(0, frame_idx - rolling_window + 1)
        t_window = ts_ds[start : frame_idx + 1]
        Z = pdf_matrix[start : frame_idx + 1].T
        T, P = np.meshgrid(t_window, p_grid)
        ax3d.cla()
        surface = ax3d.plot_surface(T, P, Z, cmap="viridis", linewidth=0, antialiased=False, alpha=0.8)
        ax3d.set_xlabel("Time (days)")
        ax3d.set_ylabel("p")
        ax3d.set_zlabel("Density")
        ax3d.set_ylim(0.0, 1.0)
        ax3d.view_init(elev=camera_elev, azim=camera_azim)
        return surface

    def init():
        _update_surface(0)
        mid_line.set_data([], [])
        cap_line.set_data([], [])
        return mid_line, cap_line

    def animate(frame_no: int):
        _update_surface(frame_no)
        t_slice = ts_ds[: frame_no + 1]
        mid_slice = mid_ds[: frame_no + 1]
        cap_slice = cap_ds[: frame_no + 1]
        cap_masked = np.where(t_slice >= cutoff_time, cap_slice, np.nan)
        mid_line.set_data(t_slice, mid_slice)
        cap_line.set_data(t_slice, cap_masked)
        ax2d.relim()
        ax2d.autoscale_view()
        return mid_line, cap_line

    frames = len(ts_ds)
    save_start = time.time()
    log.info(
        "Dual animation: begin save to %s (frames=%d, stride=%d, cutoff_ds_idx=%d)",
        save_path,
        frames,
        stride,
        cutoff_ds_idx,
    )
    try:
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2000)
        with writer.saving(fig, save_path, dpi=120):
            for i in range(frames):
                animate(i)
                writer.grab_frame()
                if (i + 1) % 50 == 0:
                    log.info("Dual animation: saved %d/%d frames (elapsed=%.1fs)", i + 1, frames, time.time() - save_start)
    except Exception as exc:  # pragma: no cover
        log.exception("Dual animation save failed: %s", exc)
    else:
        size = None
        try:
            size = os.path.getsize(save_path)
        except OSError:
            size = None
        log.info(
            "Dual animation: end save to %s (seconds=%.2f, size_bytes=%s)",
            save_path,
            time.time() - save_start,
            size,
        )
    finally:
        plt.close(fig)


def make_live_animation(
    mid_prices: Sequence[float],
    regimes: Sequence[int],
    posterior_means: Sequence[float],
    weights: Sequence[float],
    save_path: str,
    fps: int = 30,
    live: bool = False,
    logger: logging.Logger | None = None,
    max_frames: int | None = 800,
    shading_stride: int = 10,
    shading_lookback: int = 2000,
    save_timeout: float = 120.0,
):
    """
    Create and save the live animation. The save loop is bounded in wall-clock
    time to prevent hangs when ffmpeg is slow or unavailable.
    """
    import os
    import time

    import imageio_ffmpeg
    import matplotlib

    if live:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    # Always force ffmpeg discovery from imageio_ffmpeg.
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["animation.writer"] = "ffmpeg"

    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.animation import FFMpegWriter

    log = logger or logging.getLogger(__name__)
    orig_n = len(mid_prices)
    if orig_n == 0:
        log.warning("No mid prices provided; skipping animation save to %s", save_path)
        return

    target_points = 20000
    plot_stride = max(1, int(np.ceil(orig_n / target_points)))

    def _downsample(seq: Sequence[float]) -> np.ndarray:
        arr = np.asarray(seq)
        if plot_stride <= 1 or arr.size <= 1:
            return arr
        arr_ds = arr[::plot_stride]
        if arr_ds.size == 0:
            return arr
        if arr_ds[-1] != arr[-1]:
            arr_ds = np.append(arr_ds, arr[-1])
        return arr_ds

    mid_arr = _downsample(mid_prices)
    post_arr = _downsample(posterior_means)
    weight_arr = _downsample(weights)
    regime_arr_full = np.array(regimes[:orig_n], dtype=int) if regimes else np.zeros(orig_n, dtype=int)
    regime_arr_full = regime_arr_full[::plot_stride] if regime_arr_full.size else np.zeros(len(mid_arr), dtype=int)

    min_len = min(len(mid_arr), len(post_arr), len(weight_arr), len(regime_arr_full))
    mid_arr = mid_arr[:min_len]
    post_arr = post_arr[:min_len]
    weight_arr = weight_arr[:min_len]
    regime_arr_full = regime_arr_full[:min_len]

    n = min_len
    if n == 0:
        log.warning("Animation inputs empty after downsample; skipping animation save to %s", save_path)
        return
    max_frames = min(max_frames or 800, 1500)
    stride = max(1, int(np.ceil(n / max_frames)))
    frame_idx = np.arange(0, n, stride, dtype=int)
    if len(frame_idx) == 0 or frame_idx[-1] != n - 1:
        frame_idx = np.append(frame_idx, n - 1)

    frames = len(frame_idx)
    x = np.arange(n)

    log.info(
        "Animation downsample: original_n=%d, plot_stride=%d, plotted_n=%d",
        orig_n,
        plot_stride,
        n,
    )
    log.info(
        "Animation setup: n_events=%d, frames=%d, stride=%d, fps=%d, max_frames=%d",
        orig_n,
        frames,
        stride,
        fps,
        max_frames,
    )
    regime_segments: List[tuple[int, int, int]] = []
    if regime_arr_full.size:
        for r in np.unique(regime_arr_full):
            mask = regime_arr_full == r
            if mask.any():
                for s, e in _contiguous_segments(np.where(mask)[0]):
                    regime_segments.append((r, int(s), int(e)))
    regime_segments.sort(key=lambda t: t[1])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_price, ax_post, ax_weight = axes

    price_line, = ax_price.plot([], [], lw=1.5, color="steelblue")
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

    shading_artists: list = []

    def init():
        price_line.set_data([], [])
        post_line.set_data([], [])
        weight_line.set_data([], [])
        return price_line, post_line, weight_line

    def animate(frame_no: int):
        idx = int(frame_idx[min(frame_no, frames - 1)]) + 1
        price_line.set_data(x[:idx], mid_arr[:idx])
        post_line.set_data(x[:idx], post_arr[:idx])
        weight_line.set_data(x[:idx], weight_arr[:idx])

        if shading_stride and frame_no % shading_stride == 0:
            for coll in list(shading_artists):
                try:
                    coll.remove()
                except Exception:
                    pass
            shading_artists.clear()

            cutoff = max(0, idx - shading_lookback)
            for r, s, e in regime_segments:
                if e < cutoff:
                    continue
                if s >= idx:
                    break
                span_end = min(e, idx - 1)
                shading_artists.append(
                    ax_price.axvspan(s, span_end, color="lightgray", alpha=0.15 + 0.1 * (r % 3))
                )

        return price_line, post_line, weight_line

    # Keep FuncAnimation for live mode; saving is handled manually below.
    anim = None
    if live:
        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=frames, interval=1000 / fps, blit=False, save_count=frames
        )

    save_start = time.time()
    log.info("Animation: begin save to %s (frames=%d)", save_path, frames)
    try:
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
        init()
        with writer.saving(fig, save_path, dpi=120):
            for i in range(frames):
                animate(i)
                writer.grab_frame()
                if (i + 1) % 50 == 0:
                    log.info("Animation: saved %d/%d frames (elapsed=%.1fs)", i + 1, frames, time.time() - save_start)
                if time.time() - save_start > save_timeout:
                    raise TimeoutError(f"Animation save exceeded {save_timeout} seconds; aborting video save.")

    except Exception as e:  # pragma: no cover - best-effort safeguard
        log.exception("Animation save failed; continuing without video. Error: %s", e)
    else:
        size = None
        try:
            size = os.path.getsize(save_path)
        except OSError:
            size = None
        log.info(
            "Animation: end save to %s (seconds=%.2f, size_bytes=%s)",
            save_path,
            time.time() - save_start,
            size,
        )

    if live and anim is not None:
        plt.show()  # blocking, keeps window alive
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

