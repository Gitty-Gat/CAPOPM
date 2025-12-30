# CAPOPM Codex Agent Instructions (Repo Contract)

## Canonical Spec (Source of Truth)
- Implement CAPOPM exactly per the canonical PDF at:
  `docs/CAPOPM_paper.pdf`
- Do not alter theoretical math, definitions, assumptions, or phase structure.
- When implementing key equations/corrections/metrics, add brief code comments referencing the Phase/section name in the PDF (page refs if available).

## Data Scope
- Synthetic data only (no real markets): simulate populations, signals, outcomes, and dynamic market timeline.
- Transaction-level trades must be simulated (timestamped, sized orders).
- Market is dynamic in time; simulate evolving parimutuel pools and implied odds during the window (not just end totals).
- Network restricted: do not fetch external data or packages at runtime.

## Required Implementation (Phases 1–6)
Implement Phases 1–6 computationally, preserving conjugacy where required:
1) Structural prior
2) Hybrid structural + ML prior (ML prior must be synthetic and controllable: bias/variance/calibration knobs)
3) Trader information structure and behaviors (heterogeneous trader types; trade sizes vary by type)
4) Bayesian update (e.g., Beta–Binomial) exactly as in paper
5) Mapping from transaction-level trades to “evidence” / likelihood inputs
6) Full two-stage corrections: behavioral + structural (liquidity/whale/imbalance)

### Implementation Disambiguation Rules (to prevent guessing)

- Phase 1 q_str: specify either (A) Fourier inversion via paper’s characteristic function / Riccati–Volterra system,
  or (B) surrogate-only; any surrogate must be labeled and parameterized by the Phase 1 inputs.
  If structural computation is infeasible, implement a SURROGATE with the same interface and label it everywhere.

- Phase 6 Stage 1: w_beh must reweight trade sizes into effective counts while ensuring 0 ≤ y^(1) ≤ n^(1).

- Phase 6 Stage 2: define whether offsets/regimes act on (y,n), (α,β), or likelihoods; mixture must return a valid posterior distribution and valid summarized moments.

- Metrics: primary evaluation uses the final (market-close) posterior; optional time-path metrics must be reported separately.

- Statistical tests: paired by run seeds; multiple-comparisons correction is applied within each scenario × metric across model comparisons.

## Underspecified Items — Implementation Choices
- The paper leaves several definitions open; the following defaults are used to make the      implementation operational while preserving Beta–Binomial conjugacy, Phase 6 admissibility (positive pseudo-counts), and the paper’s intent. These are implementation choices, not theoretical modifications.

- Mean absolute probability error (Phase 7). 
Defined as |p_true − π_adj^YES|.

- Calibration error (Phase 7). 
Use Expected Calibration Error (ECE) with equal-width bins on [0,1]. Reliability diagrams use the same binning.

- Interval coverage (Phase 7)
Use central Beta credible intervals at 90% and 95%. Report coverage of the realized outcome A; optionally report coverage with respect to p_true.

- Stage 1 behavioral weights w_beh (Phase 6)
Use a bounded positive weight family:
w = clip(w_min, w_max, w_LS(p_mkt) × w_H(streak))
where w_LS downweights long-shot probabilities (small p_mkt) and w_H downweights herding (long streaks of identical recent orders). Bounds ensure positivity and preserve conjugacy.

- Regime likelihood L_r(D2) (Phase 6 Stage 2 mixture)
Use the Beta–Binomial marginal likelihood based on effective counts (y^(1), n^(1)) as the regime likelihood in π_r(D2).

- HMM regime dynamics (Phase 6 optional)
OFF by default. If enabled, use a user-specified transition matrix and emissions derived from L_r(D2_t); outputs feed time-varying mixture weights while preserving admissibility (α_r(t) > 0, β_r(t) > 0).


## Baselines
- Implement all baselines mentioned in the paper plus simple shrinkage priors (e.g., Beta(1,1), Beta(0.5,0.5)).
- Include at least: structural-only, ML-only, uncorrected posterior, raw parimutuel implied probability, shrinkage priors.
- Explicitly document what each baseline may “know.”
- Any baseline that uses truth-level simulated features must be labeled “ORACLE” in outputs and must not be used to claim superiority.

## Metrics (Phase 7)
- Implement every metric listed in Phase 7 of the PDF.
- Compute per-run, aggregate by scenario, and output tables + figures.
- Include calibration, proper scoring (Brier/log), discrimination, interval/coverage (if applicable), and robustness/stability as specified by the paper.

## Statistical Testing
- Required tests (paired by shared seeds across models):
  - paired t-tests (at least Brier; extend where appropriate)
  - nonparametric paired tests (e.g., Wilcoxon signed-rank)
  - bootstrap confidence intervals for mean differences
  - multiple comparison corrections: Holm and Bonferroni
- One experiment config must use one deterministic global seed. All stochastic components must be derived from that seed (numpy Generator streams are allowed).

## Experiment Design
Config-driven scenarios with fixed global seed per experiment. Include:
- Well-specified scenario (assumptions hold)
- Miscalibration: structural only / ML only / both
- Behavioral stress: herding, long-shot bias sweeps, strategic timing
- Structural stress: whale dominance, liquidity / participation imbalance
- Parameter sweeps of key knobs (α, λ, η, correction strengths, etc.)

## Outputs
- Figures, tables, simulation logs (plain text OK), reproducible scripts and notebook.
- Single master Jupyter notebook runs end-to-end:
  seed → scenarios → simulate → fit CAPOPM + baselines → metrics → stats → write artifacts.

## Engineering + Repo Practices
- Memory-safe for ~16 GB RAM; avoid huge in-memory objects; write run-level outputs to disk when needed.
- Repository structure: reusable Python modules (simulation, inference, baselines, eval/stats), results dir, configs, README with run/repro instructions.
- Do not refactor earlier phases while implementing later phases unless explicitly instructed.
- Avoid destructive git commands. Use `rg` for searching.

## Claims Policy
- No claims of general superiority.
- Results must be framed strictly as “partial validation under controlled synthetic assumptions.”
