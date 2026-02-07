"""Visualization utilities for the flat-prior synthetic reel."""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    out: List[Tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for v in idx[1:]:
        i = int(v)
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = i
        prev = i
    out.append((start, prev))
    return out


def _beta_pdf_grid(alpha: float, beta: float, p_grid: np.ndarray) -> np.ndarray:
    alpha = max(float(alpha), 1e-6)
    beta = max(float(beta), 1e-6)
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_pdf = (alpha - 1.0) * np.log(p_grid) + (beta - 1.0) * np.log(1.0 - p_grid) - log_beta
    log_pdf = log_pdf - np.max(log_pdf)
    pdf = np.exp(log_pdf)
    norm = np.trapz(pdf, p_grid)
    if norm <= 0.0:
        return np.full_like(p_grid, 1.0 / len(p_grid))
    return pdf / norm


def _downsample_indices(n: int, max_frames: int) -> np.ndarray:
    n = int(max(1, n))
    max_frames = int(min(max(10, max_frames), 900))
    stride = max(1, int(math.ceil(n / max_frames)))
    idx = np.arange(0, n, stride, dtype=np.int32)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def make_capopm_v2_animation(
    full_days: Sequence[float],
    full_prices: Sequence[float],
    incoming_days: Sequence[float],
    alpha_incoming: Sequence[float],
    beta_incoming: Sequence[float],
    training_cutoff_day: float,
    correction_active: Sequence[bool],
    correction_strength: Sequence[float],
    stage1_strength: Sequence[float],
    stage2_strength: Sequence[float],
    save_path: str,
    fps: int = 30,
    max_frames: int = 900,
    posterior_grid_points: int = 140,
    posterior_rolling_window: int = 120,
    camera_elev: float = 28.0,
    camera_azim: float = -55.0,
    logger: logging.Logger | None = None,
) -> None:
    """Create the required v2 MP4 with evolving 3D posterior and regime shading."""

    import imageio_ffmpeg
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    from matplotlib import cm
    import matplotlib.pyplot as plt

    log = logger or logging.getLogger(__name__)
    matplotlib.use("Agg")
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["animation.writer"] = "ffmpeg"
    matplotlib.rcParams["font.size"] = 10

    days_full = np.asarray(full_days, dtype=np.float64)
    prices = np.asarray(full_prices, dtype=np.float64)
    days_in = np.asarray(incoming_days, dtype=np.float64)
    alpha_arr = np.asarray(alpha_incoming, dtype=np.float64)
    beta_arr = np.asarray(beta_incoming, dtype=np.float64)
    corr_active = np.asarray(correction_active, dtype=bool)
    corr_strength = np.asarray(correction_strength, dtype=np.float64)
    s1 = np.asarray(stage1_strength, dtype=np.float64)
    s2 = np.asarray(stage2_strength, dtype=np.float64)

    n_incoming = min(len(days_in), len(alpha_arr), len(beta_arr))
    n_full = min(len(days_full), len(prices), len(corr_active), len(corr_strength), len(s1), len(s2))
    if n_incoming <= 1 or n_full <= 1:
        raise ValueError("Insufficient data to render capopm_v2 animation")

    days_in = days_in[:n_incoming]
    alpha_arr = np.clip(alpha_arr[:n_incoming], 1e-3, 1e6)
    beta_arr = np.clip(beta_arr[:n_incoming], 1e-3, 1e6)

    days_full = days_full[:n_full]
    prices = prices[:n_full]
    corr_active = corr_active[:n_full]
    corr_strength = np.clip(corr_strength[:n_full], 0.0, 1.0)
    s1 = np.clip(s1[:n_full], 0.0, 1.0)
    s2 = np.clip(s2[:n_full], 0.0, 1.0)

    p_grid = np.linspace(0.01, 0.99, int(max(40, posterior_grid_points)))
    pdf_matrix = np.vstack([_beta_pdf_grid(a, b, p_grid) for a, b in zip(alpha_arr, beta_arr)])
    z_upper = float(np.percentile(pdf_matrix, 99.5))

    frame_idx = _downsample_indices(n_incoming, max_frames=max_frames)
    days_in_ds = days_in[frame_idx]
    pdf_ds = pdf_matrix[frame_idx]
    frame_to_full = np.clip(np.searchsorted(days_full, days_in_ds, side="left"), 0, n_full - 1)
    frames = len(frame_idx)

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.08)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    axp = fig.add_subplot(gs[1])
    fig.patch.set_facecolor("#f7f7f5")
    axp.set_facecolor("#fbfaf8")

    # Static background regime patches; alpha encodes correction strength.
    stage1_mask = corr_active & (s1 >= s2)
    stage2_mask = corr_active & (s2 > s1)
    for start, end in _contiguous_segments(stage1_mask):
        alpha_span = float(0.10 + 0.35 * np.nanmean(corr_strength[start : end + 1]))
        axp.axvspan(days_full[start], days_full[end], color="#7aa67a", alpha=alpha_span, lw=0)
    for start, end in _contiguous_segments(stage2_mask):
        alpha_span = float(0.10 + 0.35 * np.nanmean(corr_strength[start : end + 1]))
        axp.axvspan(days_full[start], days_full[end], color="#c69473", alpha=alpha_span, lw=0)

    line_price, = axp.plot([], [], color="#1f4f5f", lw=1.9, label="Synthetic mid price")
    line_cutoff = axp.axvline(training_cutoff_day, color="#444444", lw=1.2, ls="--", alpha=0.8)
    axp.text(
        training_cutoff_day + 3.0,
        float(np.nanmax(prices)) * 0.985,
        "Training ends -> incoming trades + corrections",
        fontsize=9,
        color="#333333",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.18", "fc": "#f0efe9", "ec": "#d2d0c7", "alpha": 0.9},
    )
    stage1_proxy = axp.axvspan(days_full[0], days_full[0], color="#7aa67a", alpha=0.20, label="Stage1-dominant correction regime")
    stage2_proxy = axp.axvspan(days_full[0], days_full[0], color="#c69473", alpha=0.20, label="Stage2-dominant correction regime")
    axp.legend(handles=[line_price, stage1_proxy, stage2_proxy], loc="upper left", framealpha=0.95, fontsize=9)
    axp.set_xlim(float(days_full.min()), float(days_full.max()))
    price_pad = max(0.5, 0.08 * float(np.nanstd(prices)))
    axp.set_ylim(float(np.nanmin(prices) - price_pad), float(np.nanmax(prices) + price_pad))
    axp.set_xlabel("Day")
    axp.set_ylabel("Price ($)")
    axp.grid(True, alpha=0.18, lw=0.6)
    cursor = axp.axvline(days_full[0], color="#304ffe", lw=0.9, alpha=0.55)
    _ = line_cutoff  # keep explicit for readability

    ax3d.set_facecolor("#f6f4ef")
    ax3d.view_init(elev=float(camera_elev), azim=float(camera_azim))
    ax3d.set_xlim(float(days_in_ds.min()), float(days_in_ds.max()))
    ax3d.set_ylim(0.0, 1.0)
    ax3d.set_zlim(0.0, max(0.1, z_upper * 1.05))
    ax3d.set_xlabel("Incoming day")
    ax3d.set_ylabel("p")
    ax3d.set_zlabel("f_t(p)")
    ax3d.xaxis.pane.set_alpha(0.08)
    ax3d.yaxis.pane.set_alpha(0.08)
    ax3d.zaxis.pane.set_alpha(0.08)
    ax3d.set_title("CAPOPM posterior landscape (incoming window)")

    surface = None

    def draw_frame(i: int) -> None:
        nonlocal surface
        in_idx = int(i)
        start = max(0, in_idx - int(max(4, posterior_rolling_window)) + 1)
        t = days_in_ds[start : in_idx + 1]
        z = pdf_ds[start : in_idx + 1]
        tt, pp = np.meshgrid(t, p_grid)
        zz = z.T

        if surface is not None:
            try:
                surface.remove()
            except Exception:
                pass
        surface = ax3d.plot_surface(
            tt,
            pp,
            zz,
            cmap=cm.cividis,
            linewidth=0,
            antialiased=True,
            alpha=0.93,
            rcount=min(zz.shape[0], 180),
            ccount=min(zz.shape[1], 220),
        )

        fidx = int(frame_to_full[in_idx])
        line_price.set_data(days_full[: fidx + 1], prices[: fidx + 1])
        cursor.set_xdata([days_full[fidx], days_full[fidx]])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = FFMpegWriter(fps=int(fps), codec="libx264", bitrate=3200)
    save_start = os.path.getmtime(save_path) if os.path.exists(save_path) else None
    log.info("capopm_v2 animation save start: path=%s frames=%d fps=%d", save_path, frames, fps)
    with writer.saving(fig, save_path, dpi=120):
        for i in range(frames):
            draw_frame(i)
            writer.grab_frame()
            if (i + 1) % 60 == 0:
                log.info("capopm_v2 animation progress: %d/%d frames", i + 1, frames)
    plt.close(fig)
    size = os.path.getsize(save_path) if os.path.exists(save_path) else -1
    log.info(
        "capopm_v2 animation save end: path=%s size_bytes=%d mtime_changed=%s",
        save_path,
        size,
        bool(save_start is None or (os.path.getmtime(save_path) != save_start)),
    )


def make_dual_panel_animation(*args, **kwargs):  # pragma: no cover - legacy shim
    raise NotImplementedError("Use make_capopm_v2_animation for v2 visualization.")


def make_live_animation(*args, **kwargs):  # pragma: no cover - legacy shim
    raise NotImplementedError("Live animation is not used in the v2 pipeline.")
