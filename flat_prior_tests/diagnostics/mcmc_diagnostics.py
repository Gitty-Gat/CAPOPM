"""
Minimal R-hat and ESS diagnostics for lightweight chains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_chains(chains: list[np.ndarray]) -> list[np.ndarray]:
    """Split each chain in half to reduce non-stationarity effects."""
    split = []
    for arr in chains:
        n = arr.shape[0]
        if n < 2:
            continue
        half = n // 2
        split.append(arr[:half])
        split.append(arr[half:])
    return split


def compute_split_rhat(chains: list[np.ndarray]) -> float:
    if len(chains) < 2:
        return np.nan
    split = split_chains(chains)
    m = len(split)
    n = min(len(c) for c in split)
    if n == 0 or m < 2:
        return np.nan

    split = [c[:n] for c in split]
    chain_means = np.array([c.mean() for c in split])
    chain_vars = np.array([c.var(ddof=1) for c in split])
    B = n * chain_means.var(ddof=1)
    W = chain_vars.mean()
    var_hat = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_hat / W) if W > 0 else np.nan
    return float(rhat)


def compute_effective_sample_size(chains: list[np.ndarray]) -> float:
    if len(chains) == 0:
        return np.nan
    n = min(len(c) for c in chains)
    if n == 0:
        return np.nan
    centered = [c[:n] - c[:n].mean() for c in chains]
    chain_vars = np.array([np.var(c, ddof=1) for c in centered])
    W = chain_vars.mean()
    chain_means = np.array([c[:n].mean() for c in chains])
    B = n * chain_means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    rho_hat_sum = 0.0
    max_lag = min(1000, n - 1)
    for lag in range(1, max_lag):
        acov = np.mean([np.cov(c[:-lag], c[lag:], ddof=0)[0, 1] for c in centered])
        rho = 1.0 - (W - acov) / (var_hat + 1e-12)
        if rho < 0:
            break
        rho_hat_sum += rho
    ess = len(chains) * n / (1 + 2 * rho_hat_sum)
    return float(ess)


def summarize_metric(name: str, chains: list[np.ndarray]) -> pd.DataFrame:
    rhat = compute_split_rhat(chains)
    ess = compute_effective_sample_size(chains)
    return pd.DataFrame({"metric": [name], "rhat": [rhat], "ess": [ess]})

