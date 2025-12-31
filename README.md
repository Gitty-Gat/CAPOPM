# CAPOPM Synthetic Validation (Paper Preparation)

## A. Project Overview
- CAPOPM implements the canonical Bayesian parimutuel mechanism with structural priors, ML priors, Stage 1 behavioral reweighting, and Stage 2 regime mixtures (Phases 1–6 of the paper).
- Goal: evaluate whether CAPOPM satisfies the paper’s theoretical claims under controlled synthetic DGPs before moving to real markets.
- Synthetic validation is necessary for sanity and regression checks but is not sufficient for real-market credibility.

## B. Paper Structure & Claims
- Phases: structural prior (1), hybrid prior (2), trader information/behavior (3), Bayesian update (4), trade-to-evidence mapping (5), two-stage corrections (6), metrics/stats (7).
- Theorems/Propositions (examples from audit contracts):
  - Prop 6 / Lemma 3: information efficiency vs signal quality.
  - Phase 4 consistency: faster convergence with liquidity.
  - Theorem 12 / Prop 9: robustness to strategic timing via Stage 1+2.
  - Theorem 14: corrections do not increase regret.
  - Theorem 7: variance/bias decay.
  - Prop 8: regret robustness under misspecification.
  - Theorem 15: regime posterior concentration.
  - Theorem 13: projection effect on arbitrage.
- Empirical validation of any claim is permitted only when audit gating passes (see Section D).

## C. Synthetic Experiment Map (Tiers A–D)
- Tier A (mechanism):  
  - A1 INFO_EFFICIENCY: tests info aggregation; metrics: Brier/log, calibration, variance, regret vs raw.  
  - A2 TIME_TO_CONVERGE: tests variance decay vs liquidity/arrivals; metrics: time-to-eps, var slope.  
  - A3 STRATEGIC_TIMING: tests Stage1/2 protection vs late manipulation; metrics: regret_log, regime weights.
- Tier B (theoretical consequences):  
  - B1 CORRECTION_NO_REGRET (Thm 14): regret_brier/log_bad vs uncorrected.  
  - B2 ASYMPTOTIC_RATE (Thm 7): variance/bias slopes vs n.  
  - B3 MISSPEC_REGRET (Prop 8): regret surfaces across structural/ML misspec grids.  
  - B4 REGIME_CONCENTRATION (Thm 15): entropy/max-weight vs evidence.  
  - B5 ARBITRAGE_PROJECTION (Thm 13): projection distance and score deltas.
- Tier C/D scaffolding is present via registry/audit artifacts; paper-grade runs pending.
- Current empirical status (audit-driven, synthetic): **all experiments are “Not yet empirically validated”** for paper because paper-ready gates are not satisfied (low-n and/or grid_missing). Smoke evidence exists but is not claim-bearing.
  - Use wording: “Under synthetic data, current runs are smoke-level only; paper validation awaits `run_paper_suite.py` with paper thresholds.”

## D. Audit & Rigor Guarantees
- Audit gates (see `audit.json`):
  - Calibration interpretability: min unique predictions/nonempty bins/samples; otherwise ECE marked NOT_INTERPRETABLE and discrete calibration reported.
  - Coverage slicing: overall + extreme-p with support counts and flags for off-nominal coverage.
  - Effect sizes: CAPOPM vs baselines with bootstrap CI and Holm-corrected paired tests.
  - Criteria evaluation: single source of truth for pass/fail; semantics mismatch detection; grid_missing reasons recorded.
  - Reproducibility: config snapshots + hashes, registry/code hashes, reproduce_line, manifest (`PAPER_RUN_MANIFEST.json`).
- No claim may bypass these gates; if flags remain, claim is “Not yet empirically validated.”

## E. Limitations
- Synthetic DGP ≠ real markets; no external data used.
- Current artifacts are smoke-level: n_runs < paper thresholds; many degenerate calibration cases; extreme-p support sparse.
- Discrete calibration replaces ECE when bins/unique counts fail interpretability gates.
- Known borderline cases are cataloged in `results/paper_artifacts/borderline_atlas.md`.

## F. Current Status Summary
- Validated under synthetic data (paper gates passed): **none** (paper suite not yet run at required n/grid; audits show low-n/degenerate calibration/coverage flags).
- Pending paper-suite reruns: A1–A3, B1–B5 (run `python run_paper_suite.py --experiment <ID> --runs 30`).
- Awaiting real-data validation: all claims.

## G. Future Work
- Real-data validation plan (post-synthetic):
  - Data: exchange-sourced L3 equity order book data; public price/volume via `yfinance`.
  - Tests: market microstructure realism, non-stationarity stress, strategic trader behavior in real venues.
  - Principle: “Synthetic validation is necessary but insufficient for real-market credibility.”

## Artifact Pointers
- Audit outputs: `results/<scenario>/audit.json`
- Aggregated metrics/tests: `results/<scenario>/metrics_aggregated.csv`, `tests.csv`
- Manifest: `PAPER_RUN_MANIFEST.json`
- Tables: `results/paper_artifacts/claim_table.md`, `borderline_atlas.md`
- Harness: `run_paper_suite.py` (paper grids & run counts)
