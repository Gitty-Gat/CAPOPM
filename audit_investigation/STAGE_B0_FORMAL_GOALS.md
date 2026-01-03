# Stage B.0 Formal Goals

## Purpose and Scope
This document binds Stage B.1 implementation to paper-faithful, audit-faithful objectives for experiments A1, A3, B1–B5. Authority order: CAPOPM.pdf > requirements.txt > claim_table.md > all code/audits. No math reinterpretation is permitted.

## What CAPOPM Is Attempting to Establish (Paper-Cited)
- **Posterior predictive correctness (Phase 4)**: Posterior mean equals Beta-Binomial predictive probability (CAPOPM.pdf Proposition 6; cited in A1 report `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md:29`).
- **No-arbitrage preservation under kernel/regularization (Phase 7)**: Kernel adjustments preserve arbitrage-free pricing (CAPOPM.pdf Theorem 12; A3 mapping `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md:23`).
- **Correction robustness vs multimodality limits (Phase 6)**: Single-Beta approximations cannot uniformly represent multimodal mixtures (CAPOPM.pdf Theorem 14; B1 mapping `audit_investigation/B1.CORRECTION_NO_REGRET.md:35`).
- **Continuity under misspecification (Phase 4/5)**: Posterior prices are continuous in counts/parameters (CAPOPM.pdf Proposition 8; B3 mapping `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:11`).
- **Posterior concentration of mixture regimes (Phase 6/8)**: Evidence should drive regime weights toward concentration (CAPOPM.pdf Theorem 15; B4 mapping `audit_investigation/B4.REGIME_POSTERIOR_CONCENTRATION.md:29`).
- **Arbitrage-free projection validity (Phase 7)**: Projection should enforce simplex coherence without degrading scores beyond paper bounds (CAPOPM.pdf Theorem 13; B5 mapping `audit_investigation/B5.ARBITRAGE_PROJECTION_IMPACT.md:17`).
- **Asymptotic behavior with growing information (Phase 3/4/8)**: Bias/variance rates align with asymptotic pooling/unraveling statements (CAPOPM.pdf Theorem 7; B2 mapping `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md:25`).

## What CAPOPM Explicitly Does NOT Claim
- **No universal dominance over baselines**: Paper does not assert CAPOPM Brier/log-score dominance vs raw or uncorrected baselines (A1, B1, B3 reports: `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md:29`, `audit_investigation/B1.CORRECTION_NO_REGRET.md:35`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:11`).
- **No finite-sample entropy collapse**: Paper’s regime concentration (Theorem 15) does not claim entropy ≤ 0 at finite n (B4 report `audit_investigation/B4.REGIME_POSTERIOR_CONCENTRATION.md:29`).
- **No finite-sample asymptotic slope guarantees**: Theorem 7 does not claim negative bias slopes at small n_total grids (B2 report `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md:25`).
- **No guarantee of log-regret positivity under no-attack regimes**: Theorem 12 does not imply positive log-regret when attack_strength=0 (A3 report `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md:23`).

## Admissible Market Regimes (Paper-Aligned)
- **Binary parimutuel contracts** with YES/NO payoff and posterior predictive pricing (Phase 4; Proposition 6).
- **Behavioral distortions bounded by Stage1 weights** (herding/longshot within clipping bounds; Stage1 spec `src/capopm/corrections/stage1_behavioral.py:62-115` cited in upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:31-36`).
- **Structural offsets within configured regime set** (Stage2 mixture offsets; `src/capopm/corrections/stage2_structural.py:144-199`, upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:31-36`).
- **Asymptotic regimes requiring growing n_total** for rate claims (Theorem 7; B2 report `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md:25`).

## Inadmissible Market Regimes (Explicit Exclusions)
- **Strong herding or fast regime switching beyond paper assumptions** (CAPOPM.pdf impossibility results; upstream audit flag on dependence `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:65-72`).
- **Multimodal belief states forced into single-Beta without mixture mode enabled** (Theorem 14 limitations; B1/B3 reports).
- **Non-binary payoff structures or multi-asset baskets** (outside Phase 4 binary Beta-Binomial).
- **Real-time adversaries exceeding modeled attack structure** (A3 uses bounded attack_strength/window/scale; other adversary models are out of scope).

## Trader Behavior and Information Flow Assumptions
- **Price-taking, risk-neutral traders with Bernoulli signals** (Trader model assumptions noted in upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:4-10`; aligns with Phase 3).
- **Signal quality encoded via Beta parameters and arrival process** (A1 sweep; `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md:4-8`).
- **Behavioral adjustments via Stage1 weights; structural adjustments via Stage2 mixture offsets** (upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:31-36`).
- **Conditional independence approximation for likelihood (Beta-Binomial)** (upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:22-28`).

## Failure Conditions (Expected Breakpoints)
- **Insufficient grid/runs to meet paper asymptotic/finite-sample requirements** (systemic SD-03 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:41-43`).
- **Audit over-claims vs paper (dominance, entropy=0) remain uncorrected** (systemic SD-02 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:37-40`).
- **Grid aggregation not aligned to paper sweeps** leading to perpetual indeterminates (systemic SD-01 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:34-36`).
- **Structural prior surrogate deviates from Phase 1 math** (systemic SD-06 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:48-50`; upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:12-19`).
- **Silent fallbacks/clamping/clipping masking evidence** (systemic SD-05 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:44-47`; upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:31-36,49-55`).

## Stage B.1 Readiness Statement
Given Stage A closure under §2.2 and the above formal goals, Stage B.1 may proceed only under these governance constraints: paper alignment enforced, no reinterpretation, explicit handling of systemic defects (SD-01–SD-06), and adherence to requirements.txt as authoritative spec input.
