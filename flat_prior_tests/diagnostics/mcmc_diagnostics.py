"""
Minimal split R-hat and ESS diagnostics for lightweight chains.
Intended to be robust to tiny chains and return NaN when undefined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_chains(chains: list[np.ndarray]) -> list[np.ndarray]:
    split: list[np.ndarray] = []
    for arr in chains:
        n = int(arr.shape[0])
        if n < 4:  # need at least 2 per half to use ddof=1 safely
            continue
        half = n // 2
        a = arr[:half]
        b = arr[half:]
        if len(a) >= 2 and len(b) >= 2:
            split.append(a)
            split.append(b)
    return split


def compute_split_rhat(chains: list[np.ndarray]) -> float:
    split = split_chains(chains)
    m = len(split)
    if m < 2:
        return float("nan")

    n = min(len(c) for c in split)
    if n < 2:
        return float("nan")

    split = [c[:n] for c in split]

    chain_means = np.array([c.mean() for c in split], dtype=float)
    chain_vars = np.array([c.var(ddof=1) for c in split], dtype=float)

    W = float(chain_vars.mean())
    if not np.isfinite(W) or W <= 0.0:
        # If within-chain variance is zero, R-hat is not informative; return 1.0 by convention
        return 1.0

    B = float(n * chain_means.var(ddof=1)) if m >= 2 else 0.0

    var_hat = (n - 1) / n * W + B / n
    if not np.isfinite(var_hat) or var_hat <= 0.0:
        return float("nan")

    return float(np.sqrt(var_hat / W))


def compute_effective_sample_size(chains: list[np.ndarray]) -> float:
    if len(chains) < 1:
        return float("nan")

    n = min(len(c) for c in chains)
    if n < 3:
        return float("nan")

    # truncate to common length
    chains_n = [c[:n].astype(float) for c in chains]
    centered = [c - c.mean() for c in chains_n]

    chain_vars = np.array([np.var(c, ddof=1) for c in centered], dtype=float)
    W = float(chain_vars.mean())
    if not np.isfinite(W) or W <= 0.0:
        return float(len(chains) * n)

    chain_means = np.array([c.mean() for c in chains_n], dtype=float)
    B = float(n * chain_means.var(ddof=1)) if len(chains) >= 2 else 0.0

    var_hat = (n - 1) / n * W + B / n
    if not np.isfinite(var_hat) or var_hat <= 0.0:
        return float("nan")

    rho_hat_sum = 0.0
    max_lag = min(200, n - 1)  # keep it cheap

    for lag in range(1, max_lag + 1):
        # average autocovariance across chains
        acovs = []
        for c in centered:
            x = c[:-lag]
            y = c[lag:]
            if len(x) < 2:
                continue
            acovs.append(float(np.mean((x - x.mean()) * (y - y.mean()))))
        if not acovs:
            break

        acov = float(np.mean(acovs))
        rho = acov / (var_hat + 1e-12)
        if rho < 0:
            break
        rho_hat_sum += rho

    ess = len(chains) * n / (1.0 + 2.0 * rho_hat_sum)
    return float(max(1.0, ess))


def summarize_metric(name: str, chains: list[np.ndarray]) -> pd.DataFrame:
    rhat = compute_split_rhat(chains)
    ess = compute_effective_sample_size(chains)
    return pd.DataFrame({"metric": [name], "rhat": [rhat], "ess": [ess]})
