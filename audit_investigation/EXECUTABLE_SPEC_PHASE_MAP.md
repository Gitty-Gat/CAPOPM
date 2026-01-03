# Executable Spec Phase Map (Stage B.0)

Authority: CAPOPM.pdf phases; requirements.txt; claim_table.md. Each phase maps paper intent → modules → inputs → outputs → invariants → audit touchpoints. Invariants must be enforced in Stage B.1 without reinterpretation.

## Phase Map
| Paper Phase | Module(s) | Inputs | Outputs | Invariants | Audit Touchpoints |
| --- | --- | --- | --- | --- | --- |
| Phase 1: Structural Prior | `src/capopm/structural_prior.py:18-70` | Structural params (T, K, S0, V0, moneyness), jitter | `q_str` ∈ (0,1) | Positivity of params; monotone tail sensitivity; surrogate fidelity gap noted (SD-06) | Implicit in all posterior metrics; prior fidelity affects A1/B1/B3/B4/B5 (upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:12-19`) |
| Phase 2: ML Prior & Hybrid Fusion | `src/capopm/ml_prior.py:15-55`; `src/capopm/hybrid_prior.py:13-55` | ML predictions (`p_ml`), pseudo-counts (`n_ml`, `r_ml`), structural counts | Beta parameters α0, β0; weights | α0, β0 > 0; weights sum nonnegative; p_ml clipped to (1e-12, 1-1+eps) | A1/B1/B3 baselines; continuity claims (Prop 8) |
| Phase 3: Trader & Market Simulation | `src/capopm/market_simulator.py:73-148`; `src/capopm/trader_model.py:44-113` | `MarketConfig`, trader list, p_true, RNG | `trade_tape`, `pool_path` | Pools > 0; trade sizes > 0; herding_intensity ∈ [0,1]; outcome ∈ {0,1} | Inputs to all experiments; coverage/calibration rely on p_true (`audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:4-10`) |
| Phase 4: Likelihood & Posterior Update | `src/capopm/likelihood.py:12-30,33-43`; `src/capopm/posterior.py:28-135`; `src/capopm/pricing.py:14-44` | Counts y, n; α0, β0; priors | α_post, β_post; posterior mean/variance; prices; credible intervals | α_post, β_post > 0; prices ∈ [0,1]; YES+NO=1; Beta-Binomial conjugacy | Proposition 6 identity (A1); coverage/calibration metrics; all regret metrics |
| Phase 5: Mixture/Multimodal Handling | `src/capopm/corrections/stage2_structural.py:144-199` (mixture path) | Regime priors, offsets, counts | Regime weights, mixture mean/var | Weights ≥ 0, sum=1; params positive; no label switching; entropy ≥ 0 | B4 regime entropy/max weight; mixture continuity (Prop 9) |
| Phase 6: Behavioral & Structural Corrections | `src/capopm/corrections/stage1_behavioral.py:62-115`; `stage2_structural.py:144-199` | Trade weights (herding/longshot), regime offsets | Weighted counts; adjusted posteriors | Stage1 weights clipped [w_min, w_max]; Stage2 clamping logged; monotone weight application | B1/B3 regrets; A3 corrected vs uncorrected; SD-05 fallbacks |
| Phase 7: Projection / Arbitrage Enforcement | `src/capopm/experiments/projection_utils.py:6-78`; `pricing.py:34-44` | Probabilities (possibly violated), violation strength | Projected probabilities; distances | Probabilities nonnegative; sum to 1 after projection; projection distance ≥ 0 | B5 projection distance/delta scores; Theorem 13 alignment |
| Phase 8: Audit Layer & Aggregation | `src/capopm/experiments/runner.py:49-412,479-789`; `src/capopm/experiments/audit.py:332-418`; `src/capopm/experiments/audit_contracts.py` | Per-run metrics, configs, sweep params | Aggregated metrics, summary.json, audit.json | Mean aggregation NaN-safe; calibration requires ≥2 predictions; grid_points_observed must aggregate actual sweep; no silent fallback of gates | Grid_requirement (SD-01), calibration status in borderline_atlas, all criteria paths in claim_table.md |

## Invariant Enforcement Notes
- **Probability simplex**: All prices and projected probabilities must satisfy nonnegativity and sum-to-one (pricing, projection_utils). Tied to Theorem 13 audits (B5).
- **Positive Beta parameters**: α0, β0, α_post, β_post > 0 to ensure Proposition 6 applicability (Phase 4).
- **Weight normalization**: Stage2 regime weights sum to 1; Stage1 weights within clip bounds (SD-05).
- **Grid completeness**: grid_points_observed must reflect full sweep (SD-01) before evaluating any requires_grid criterion (A1, A3, B3, B4).
- **Run-count readiness**: paper_ready_min_runs thresholds (paper_config) must gate claims requiring finite-sample stability (SD-03).

## Audit Touchpoint Index (Scenario-Level)
- **A1**: Metrics `aggregated_metrics.capopm.brier`, grid requirement; invariant: posterior mean identity vs baseline (Proposition 6) — see `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md`.
- **A3**: `regret_log` vs uncorrected, grid over attack_strength/window/scale; invariant: arbitrage preservation, not regret dominance (Theorem 12) — `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md`.
- **B1**: `regret_brier`, `regret_log_bad` vs uncorrected; invariant: multimodal limitation, not dominance (Theorem 14) — `audit_investigation/B1.CORRECTION_NO_REGRET.md`.
- **B2**: `summary.status.metrics.rate_bias_vs_n_capopm`; invariant: asymptotic trend, not finite-sample slope sign (Theorem 7) — `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md`.
- **B3**: Regrets and grid over structural_shift × ml_bias; invariant: continuity, not dominance (Proposition 8); grid aggregation required — `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md`.
- **B4**: `regime_entropy`, evidence_strength grid; invariant: concentration trend, not entropy zero (Theorem 15) — `audit_investigation/B4.REGIME_POSTERIOR_CONCENTRATION.md`.
- **B5**: Projection distances and score deltas; invariant: simplex enforcement; no guaranteed score improvement (Theorem 13) — `audit_investigation/B5.ARBITRAGE_PROJECTION_IMPACT.md`.
