#!/usr/bin/env python3
"""
CAPOPM vs Kalshi, Phases 4–8, using separate TRAIN/TEST CSVs.

TRAIN CSV (settled markets):
    kalshi_capopm_train.csv  -> used for structural + ML priors and behavior/liquidity calibration

TEST CSV (open market):
    kalshi_capopm_test.csv   -> target KXINXY-25DEC31-B6900 bucket (no realized_outcome yet)

Pipeline:
    - Build robust structural prior from TRAIN winners.
    - Train ML prior on TRAIN only.
    - Reliability-weight the priors (Brier-based) into a hybrid Beta prior.
    - Do Beta-Binomial updates with raw and corrected tickets.
    - Evaluate performance on TRAIN (only place with outcomes).
    - Print CAPOPM vs Kalshi probabilities for the TEST row(s).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

TRAIN_CSV_PATH = "kalshi_capopm_train.csv"
TEST_CSV_PATH = "kalshi_capopm_test.csv"

MIN_TICKETS = 1
ETA_TOTAL = 30.0
CALIB_BINS = np.linspace(0.0, 1.0, 11)


# ----------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------

def normal_bucket_prob(mu: float, sigma: float, low: float, high: float) -> float:
    """P(low <= X < high) for X ~ N(mu, sigma^2)."""
    if sigma <= 0:
        return 1.0e-6

    from math import erf, sqrt

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))

    p = cdf(high) - cdf(low)
    return max(0.0, min(1.0, p))


def safe_prob(p, eps: float = 1e-6):
    """Clip probabilities into (eps, 1-eps). Works for scalars or arrays."""
    return np.clip(p, eps, 1.0 - eps)


# ----------------------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    return_target: bool = True,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Build ML feature matrix X and optionally target y from the DataFrame.

    Features:
        - mid_strike
        - width
        - total_tickets
        - yes_share
        - last_price_dollars
        - time_to_expiry_days

    If return_target is False, y is returned as None.
    """
    df = df.copy()

    df["mid_strike"] = 0.5 * (df["floor_strike"] + df["cap_strike"])
    df["width"] = df["cap_strike"] - df["floor_strike"]

    df["yes_share"] = df["yes_tickets"] / df["total_tickets"].replace(0, np.nan)
    df["yes_share"] = df["yes_share"].fillna(0.5)

    df["time_to_expiry_days"] = (df["expiration_ts"] - df["created_ts"]) / (60 * 60 * 24)

    feature_cols = [
        "mid_strike",
        "width",
        "total_tickets",
        "yes_share",
        "last_price_dollars",
        "time_to_expiry_days",
    ]

    X = df[feature_cols].to_numpy(dtype=float)

    if return_target:
        y = df["realized_outcome"].to_numpy(dtype=int)
    else:
        y = None

    return X, y


# ----------------------------------------------------------------------
# ROBUST STRUCTURAL PRIOR
# ----------------------------------------------------------------------

def estimate_spot_distribution(df_train: pd.DataFrame) -> Tuple[float, float]:
    """
    Estimate S_T distribution from TRAIN winners:

    - Take midpoints of all winning buckets.
    - Fit Normal(mu_hat, sigma_hat^2).
    """
    winners = df_train[df_train["realized_outcome"] == 1].copy()
    if winners.empty:
        # fallback
        return 6800.0, 500.0

    spot = 0.5 * (winners["floor_strike"] + winners["cap_strike"])
    mu_hat = float(spot.mean())
    sigma_hat = float(spot.std(ddof=1)) if len(spot) > 1 else 500.0
    if sigma_hat <= 0:
        sigma_hat = 500.0
    return mu_hat, sigma_hat


def structural_bucket_probs(
    df_all: pd.DataFrame, mu: float, sigma: float
) -> np.ndarray:
    lows = df_all["floor_strike"].to_numpy(dtype=float)
    highs = df_all["cap_strike"].to_numpy(dtype=float)

    q = np.array(
        [normal_bucket_prob(mu, sigma, lo, hi) for lo, hi in zip(lows, highs)],
        dtype=float,
    )
    return safe_prob(q)


# ----------------------------------------------------------------------
# ROBUST ML PRIOR
# ----------------------------------------------------------------------

@dataclass
class MLModel:
    clf: LogisticRegression


def train_ml_prior(df_train: pd.DataFrame) -> MLModel:
    """
    Train logistic regression with class_weight='balanced' on TRAIN only.
    """
    X_train, y_train = engineer_features(df_train, return_target=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    return MLModel(clf=clf)


def ml_bucket_probs(model: MLModel, df_all: pd.DataFrame) -> np.ndarray:
    X_all, _ = engineer_features(df_all, return_target=False)
    q_ml_all = model.clf.predict_proba(X_all)[:, 1]
    return safe_prob(q_ml_all)


# ----------------------------------------------------------------------
# RELIABILITY-WEIGHTED HYBRID PRIOR
# ----------------------------------------------------------------------

@dataclass
class Priors:
    alpha0: np.ndarray
    beta0: np.ndarray
    q_struct: np.ndarray
    q_ml: np.ndarray
    q_hybrid: np.ndarray
    mu: float
    sigma: float
    lambda_struct: float
    lambda_ml: float
    brier_struct: float
    brier_ml: float


def build_robust_hybrid_priors(
    df_all: pd.DataFrame,
    df_train: pd.DataFrame,
    eta_total: float = ETA_TOTAL,
) -> Priors:
    """
    Structural q_struct + ML q_ml, reliability-weighted into q_hybrid
    and turned into a Beta(α0, β0) prior.
    """
    # Structural
    mu_hat, sigma_hat = estimate_spot_distribution(df_train)
    q_struct_all = structural_bucket_probs(df_all, mu_hat, sigma_hat)

    # ML
    ml_model = train_ml_prior(df_train)
    q_ml_all = ml_bucket_probs(ml_model, df_all)

    # Reliability weights from TRAIN
    # Align TRAIN mask on df_all
    # Use whatever key you want; here assume (event_ticker, market_ticker) uniqueness
    keys_all = df_all[["event_ticker", "market_ticker"]].astype(str)
    keys_train = df_train[["event_ticker", "market_ticker"]].astype(str)

    train_mask = keys_all.merge(
        keys_train.drop_duplicates(),
        how="left",
        on=["event_ticker", "market_ticker"],
        indicator=True
    )["_merge"].eq("both").to_numpy()

    y_train = df_all.loc[train_mask, "realized_outcome"].to_numpy(dtype=int)
    q_struct_train = q_struct_all[train_mask]
    q_ml_train = q_ml_all[train_mask]

    brier_struct = brier_score_loss(y_train, safe_prob(q_struct_train))
    brier_ml = brier_score_loss(y_train, safe_prob(q_ml_train))

    w_struct = 1.0 / max(brier_struct, 1e-6)
    w_ml = 1.0 / max(brier_ml, 1e-6)
    lambda_struct = w_struct / (w_struct + w_ml)
    lambda_ml = w_ml / (w_struct + w_ml)

    q_hybrid = lambda_struct * q_struct_all + lambda_ml * q_ml_all
    q_hybrid = safe_prob(q_hybrid)

    alpha0 = eta_total * q_hybrid
    beta0 = eta_total * (1.0 - q_hybrid)

    return Priors(
        alpha0=alpha0,
        beta0=beta0,
        q_struct=q_struct_all,
        q_ml=q_ml_all,
        q_hybrid=q_hybrid,
        mu=mu_hat,
        sigma=sigma_hat,
        lambda_struct=float(lambda_struct),
        lambda_ml=float(lambda_ml),
        brier_struct=float(brier_struct),
        brier_ml=float(brier_ml),
    )


# ----------------------------------------------------------------------
# POSTERIOR UPDATES
# ----------------------------------------------------------------------

@dataclass
class PosteriorRaw:
    p_mean: np.ndarray


def update_posterior_raw(df_all: pd.DataFrame, priors: Priors) -> PosteriorRaw:
    yes = df_all["yes_tickets"].to_numpy(dtype=float)
    no = df_all["no_tickets"].to_numpy(dtype=float)

    alpha_post = priors.alpha0 + yes
    beta_post = priors.beta0 + no
    p_mean = alpha_post / (alpha_post + beta_post)

    return PosteriorRaw(p_mean=p_mean)


# ----------------------------------------------------------------------
# BEHAVIORAL + LIQUIDITY CORRECTIONS
# ----------------------------------------------------------------------

@dataclass
class CorrModels:
    price_calibrator: LogisticRegression
    median_tickets: float


def fit_behavioral_liquidity_models(df_train: pd.DataFrame) -> CorrModels:
    p_k = df_train["kalshi_implied_yes_prob"].to_numpy(dtype=float).reshape(-1, 1)
    y = df_train["realized_outcome"].to_numpy(dtype=int)

    p_k = safe_prob(p_k)
    price_cal = LogisticRegression(max_iter=1000)
    price_cal.fit(p_k, y)

    median_tickets = float(df_train["total_tickets"].median())
    return CorrModels(price_calibrator=price_cal, median_tickets=median_tickets)


@dataclass
class CorrectedCounts:
    yes_corr: np.ndarray
    no_corr: np.ndarray


def apply_corrections(df_all: pd.DataFrame, models: CorrModels) -> CorrectedCounts:
    total = df_all["total_tickets"].to_numpy(dtype=float)
    p_k = df_all["kalshi_implied_yes_prob"].to_numpy(dtype=float).reshape(-1, 1)

    p_k_clip = safe_prob(p_k)
    w = models.price_calibrator.predict_proba(p_k_clip)[:, 1]
    w = safe_prob(w)

    yes1 = total * w
    no1 = total - yes1

    median_t = max(models.median_tickets, 1.0)
    liquidity_ratio = total / median_t
    lambda_liq = np.clip(liquidity_ratio, 0.3, 1.0)

    yes_corr = lambda_liq * yes1
    no_corr = lambda_liq * no1

    return CorrectedCounts(yes_corr=yes_corr, no_corr=no_corr)


@dataclass
class PosteriorFull:
    p_mean: np.ndarray


def update_posterior_corrected(
    priors: Priors,
    counts: CorrectedCounts,
) -> PosteriorFull:
    alpha_post = priors.alpha0 + counts.yes_corr
    beta_post = priors.beta0 + counts.no_corr
    p_mean = alpha_post / (alpha_post + beta_post)
    return PosteriorFull(p_mean=p_mean)


# ----------------------------------------------------------------------
# EVALUATION ON TRAIN ONLY
# ----------------------------------------------------------------------

@dataclass
class EvaluationResults:
    brier: Dict[str, float]
    logloss: Dict[str, float]
    calib_full: pd.DataFrame


def evaluate_on_train(
    df_all: pd.DataFrame,
    df_train: pd.DataFrame,
    priors: Priors,
    post_raw: PosteriorRaw,
    post_full: PosteriorFull,
    bins: np.ndarray = CALIB_BINS,
) -> EvaluationResults:
    """
    Compute scores only on TRAIN rows (where realized_outcome is known).
    """
    keys_all = df_all[["event_ticker", "market_ticker"]].astype(str)
    keys_train = df_train[["event_ticker", "market_ticker"]].astype(str)

    mask_train = keys_all.merge(
        keys_train.drop_duplicates(),
        how="left",
        on=["event_ticker", "market_ticker"],
        indicator=True
    )["_merge"].eq("both").to_numpy()

    y = df_all.loc[mask_train, "realized_outcome"].to_numpy(dtype=int)

    p_k = df_all.loc[mask_train, "kalshi_implied_yes_prob"].to_numpy(dtype=float)
    p_prior = priors.alpha0[mask_train] / (priors.alpha0[mask_train] + priors.beta0[mask_train])
    p_raw = post_raw.p_mean[mask_train]
    p_full = post_full.p_mean[mask_train]

    p_k = safe_prob(p_k)
    p_prior = safe_prob(p_prior)
    p_raw = safe_prob(p_raw)
    p_full = safe_prob(p_full)

    brier = {
        "kalshi": brier_score_loss(y, p_k),
        "prior": brier_score_loss(y, p_prior),
        "capopm_raw": brier_score_loss(y, p_raw),
        "capopm_full": brier_score_loss(y, p_full),
    }

    logloss = {
        "kalshi": log_loss(y, p_k),
        "prior": log_loss(y, p_prior),
        "capopm_raw": log_loss(y, p_raw),
        "capopm_full": log_loss(y, p_full),
    }

    # calibration for CAPOPM full
    bin_ids = np.digitize(p_full, bins) - 1
    records = []
    for b in range(len(bins) - 1):
        mask_bin = bin_ids == b
        if not np.any(mask_bin):
            continue
        avg_pred = float(p_full[mask_bin].mean())
        emp_freq = float(y[mask_bin].mean())
        n = int(mask_bin.sum())
        records.append(
            {
                "bin_low": float(bins[b]),
                "bin_high": float(bins[b + 1]),
                "n": n,
                "avg_pred": avg_pred,
                "emp_freq": emp_freq,
            }
        )

    calib_df = pd.DataFrame(records)
    return EvaluationResults(brier=brier, logloss=logloss, calib_full=calib_df)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"TRAIN CSV not found at {TRAIN_CSV_PATH}")
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"TEST CSV not found at {TEST_CSV_PATH}")

    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)

    # Basic filtering
    df_train = df_train[(df_train["total_tickets"] >= MIN_TICKETS)].copy()
    df_train = df_train[df_train["floor_strike"].notna() & df_train["cap_strike"].notna()].copy()

    df_test = df_test[(df_test["total_tickets"] >= MIN_TICKETS)].copy()
    df_test = df_test[df_test["floor_strike"].notna() & df_test["cap_strike"].notna()].copy()

    if df_train.empty:
        raise ValueError("Training set is empty after filtering.")
    if df_test.empty:
        print("WARNING: Test set is empty after filtering.")

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    print(f"TRAIN rows: {len(df_train)}, TEST rows: {len(df_test)}, ALL rows: {len(df_all)}")

    # PHASE 4: robust priors
    print("\nBuilding robust structural + ML + hybrid priors (Phase 4)...")
    priors = build_robust_hybrid_priors(df_all, df_train, eta_total=ETA_TOTAL)
    print(f"Structural mu_hat={priors.mu:.4f}, sigma_hat={priors.sigma:.4f}")
    print(f"Hybrid weights: lambda_struct={priors.lambda_struct:.3f}, "
          f"lambda_ml={priors.lambda_ml:.3f}")
    print(f"TRAIN Brier: structural={priors.brier_struct:.6f}, ML={priors.brier_ml:.6f}")

    # PHASE 5: raw posterior
    print("\nUpdating raw posteriors from tickets (Phase 5)...")
    post_raw = update_posterior_raw(df_all, priors)

    # PHASE 6: behavior + liquidity
    print("\nFitting behavioral & liquidity models (Phase 6)...")
    corr_models = fit_behavioral_liquidity_models(df_train)
    counts_corr = apply_corrections(df_all, corr_models)
    post_full = update_posterior_corrected(priors, counts_corr)

    # PHASE 7: evaluation on TRAIN rows
    print("\nEvaluating Kalshi vs CAPOPM on TRAIN only (Phase 7)...")
    eval_res = evaluate_on_train(df_all, df_train, priors, post_raw, post_full)

    print("\n--- Brier scores (TRAIN, lower is better) ---")
    for k, v in eval_res.brier.items():
        print(f"{k:12s}: {v:.6f}")

    print("\n--- Log losses (TRAIN, lower is better) ---")
    for k, v in eval_res.logloss.items():
        print(f"{k:12s}: {v:.6f}")

    print("\n--- CAPOPM full calibration (TRAIN) ---")
    print(eval_res.calib_full.to_string(index=False))

    # PHASE 8-ish: print TEST bucket(s) with probabilities
    print("\nTEST market(s): CAPOPM vs Kalshi (no outcomes yet)")
    df_all["p_capopm_full"] = post_full.p_mean
    df_all["p_prior_mean"] = priors.alpha0 / (priors.alpha0 + priors.beta0)
    df_all["p_capopm_raw"] = post_raw.p_mean

    # Identify test rows as those not in TRAIN keys
    keys_all = df_all[["event_ticker", "market_ticker"]].astype(str)
    keys_train = df_train[["event_ticker", "market_ticker"]].astype(str)
    mask_train = keys_all.merge(
        keys_train.drop_duplicates(),
        how="left",
        on=["event_ticker", "market_ticker"],
        indicator=True
    )["_merge"].eq("both").to_numpy()

    df_test_view = df_all[~mask_train].copy()

    if df_test_view.empty:
        print("(No test rows to show.)")
    else:
        display_cols = [
            "event_ticker",
            "market_ticker",
            "floor_strike",
            "cap_strike",
            "kalshi_implied_yes_prob",
            "p_prior_mean",
            "p_capopm_raw",
            "p_capopm_full",
            "total_tickets",
        ]
        print(df_test_view[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
