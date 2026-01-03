# Stage A Closure Checklist (Scope: A1, A3, B1–B5, Upstream)

## A) Inventory of Required Stage A Artifacts
| Artifact | Absolute Path | Status |
| --- | --- | --- |
| A1.INFO_EFFICIENCY_CURVES.md | D:\CAPOPM\CAPOPM\audit_investigation\A1.INFO_EFFICIENCY_CURVES.md | Present |
| A2.TIME_TO_CONVERGE.md | D:\CAPOPM\CAPOPM\audit_investigation\A2.TIME_TO_CONVERGE.md | Excluded by governance (not required for completeness) |
| A3.STRATEGIC_TIMING_ATTACK.md | D:\CAPOPM\CAPOPM\audit_investigation\A3.STRATEGIC_TIMING_ATTACK.md | Present |
| B1.CORRECTION_NO_REGRET.md | D:\CAPOPM\CAPOPM\audit_investigation\B1.CORRECTION_NO_REGRET.md | Present |
| B2.ASYMPTOTIC_RATE_CHECK.md | D:\CAPOPM\CAPOPM\audit_investigation\B2.ASYMPTOTIC_RATE_CHECK.md | Present |
| B3.MISSPECIFICATION_REGRET_GRID.md | D:\CAPOPM\CAPOPM\audit_investigation\B3.MISSPECIFICATION_REGRET_GRID.md | Present |
| B4.REGIME_POSTERIOR_CONCENTRATION.md | D:\CAPOPM\CAPOPM\audit_investigation\B4.REGIME_POSTERIOR_CONCENTRATION.md | Present |
| B5.ARBITRAGE_PROJECTION_IMPACT.md | D:\CAPOPM\CAPOPM\audit_investigation\B5.ARBITRAGE_PROJECTION_IMPACT.md | Present |
| UPSTREAM_PIPELINE_AUDIT.md | D:\CAPOPM\CAPOPM\audit_investigation\UPSTREAM_PIPELINE_AUDIT.md | Present |

## B) Failed/Indeterminate Criteria with Root-Cause Classification
- **A1.INFO_EFFICIENCY_CURVES**
  - `capopm_dominates_raw_parimutuel_brier` (scenario `a1_info_eff_q65_inf30_adv10_seed99101`) — category (1) paper–audit semantic mismatch (Proposition 6 is posterior-mean identity, not dominance); ref `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md:29`.
  - `grid_requirement` (same scenario) — category (2) audit design/aggregation defect (per-scenario grid counting prevents evaluation); ref `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md:36`.
- **A3.STRATEGIC_TIMING_ATTACK**
  - `regret_log_non_negative` (scenarios `A3_strategic_timing__attack0__seed0__window20__scale1`, `A3_strategic_timing__attack0__seed1__window20__scale3`) — category (1) paper–audit semantic mismatch (Theorem 12 does not imply log-regret dominance); refs `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md:23`, `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md:32`.
  - `grid_requirement` (all four scenarios) — category (2) audit design/aggregation defect (grid points not aggregated across scenarios); ref `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md:40`.
- **B1.CORRECTION_NO_REGRET**
  - `regret_brier_non_positive` (scenarios `...attack0__liqhigh__seed1`, `...attack100__liqlow__seed2`) — category (1) paper–audit semantic mismatch (Theorem 14 not a regret dominance claim); ref `audit_investigation/B1.CORRECTION_NO_REGRET.md:35`.
  - `regret_log_bad_non_positive` (scenarios `...attack0__liqhigh__seed1`, `...attack100__liqhigh__seed3`) — category (1) paper–audit semantic mismatch; ref `audit_investigation/B1.CORRECTION_NO_REGRET.md:48`.
- **B2.ASYMPTOTIC_RATE_CHECK**
  - `bias_slope_negative` (scenario `B2_asymptotic_rate__seed0`) — category (1) paper–audit semantic mismatch (Theorem 7 asymptotic; no slope sign guarantee); ref `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md:25`.
- **B3.MISSPECIFICATION_REGRET_GRID**
  - `regret_brier_non_positive` (all four scenarios) — category (1) paper–audit semantic mismatch (Proposition 8 continuity vs regret dominance); refs `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:11`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:35`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:55`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:72`.
  - `regret_log_bad_non_positive` (scenarios `struct0__ml0__seed0`, `struct0__ml5__seed1`, `struct10__ml5__seed101`) — category (1) paper–audit semantic mismatch; refs `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:17`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:40`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:79`.
  - `grid_requirement` (all four scenarios) — category (2) audit design/aggregation defect (per-scenario `grid_points_observed=1`); refs `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:23`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:47`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:63`, `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:86`.
- **B4.REGIME_POSTERIOR_CONCENTRATION**
  - `regime_entropy_defined` (scenarios `evidence150__seed1`, `evidence50__seed0`) — category (1) paper–audit semantic mismatch (Theorem 15 concentration vs entropy ≤ 0); ref `audit_investigation/B4.REGIME_POSTERIOR_CONCENTRATION.md:29`.
  - `grid_requirement` (both scenarios) — category (2) audit design/aggregation defect (grid not aggregated across evidence levels); ref `audit_investigation/B4.REGIME_POSTERIOR_CONCENTRATION.md:34`.
- **B5.ARBITRAGE_PROJECTION_IMPACT**
  - No failed or indeterminate criteria; all audited criteria passed (see `audit_investigation/B5.ARBITRAGE_PROJECTION_IMPACT.md:17`).

## C) Systemic Defects Register
| ID | Description | Where Observed | Affected Claims/Criteria | Module Boundary |
| --- | --- | --- | --- | --- |
| SD-01 | Per-scenario grid counting sets `grid_points_observed=1`, causing `grid_missing_for_claim` despite multi-point sweeps. | A1 grid gate `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md:36`; A3 grid gates `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md:40`; B3 grid gates `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:23`; B4 grid gates `audit_investigation/B4.REGIME_POSTERIOR_CONCENTRATION.md:34` | All grid_requirement criteria across A1, A3, B3, B4 | `src/capopm/experiments/audit.py` grid aggregation; `paper_config.py` thresholds |
| SD-02 | Audit contracts over-claim paper results (dominance/regret/entropy) beyond theorem statements. | A1 dominance `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md:29`; A3 log-regret `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md:23`; B1 regrets `audit_investigation/B1.CORRECTION_NO_REGRET.md:35`; B2 slope sign `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md:25`; B3 regrets `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:11`; B4 entropy `audit_investigation/B4.REGIME_POSTERIOR_CONCENTRATION.md:29` | Criteria tied to Theorems 7, 12, 14, 15 and Proposition 8 mapped to regret/entropy/dominance gates | `src/capopm/experiments/audit_contracts.py` (claim mapping), audit evaluation |
| SD-03 | Finite-sample configurations far below paper-ready thresholds render claims untestable (n_runs ≤ 5 or minimal n_total grid). | A1 n_runs=4 `audit_investigation/A1.INFO_EFFICIENCY_CURVES.md:30`; A3 n_runs=5 `audit_investigation/A3.STRATEGIC_TIMING_ATTACK.md:24`; B1 n_runs=3 `audit_investigation/B1.CORRECTION_NO_REGRET.md:36`; B3 n_runs=3 `audit_investigation/B3.MISSPECIFICATION_REGRET_GRID.md:12`; B4 n_runs=5 `audit_investigation/B4.REGIME_POSTERIOR_CONCENTRATION.md:30`; B2 uses three n_total points `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md:26` | Dominance/regret/grid/entropy/rate claims across A1–B4 | Experiment configs and runner aggregation (n_runs), `paper_config.py` min-run thresholds |
| SD-04 | Missing implementation visibility for B2 experiment logic (rate computation opaque). | `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md:22` | B2 rate metrics (`rate_bias_vs_n_capopm`, related slope criteria) | Missing/unknown `src/capopm/experiments/b2_asymptotic_rate_check.py` (experiment layer) |
| SD-05 | Silent fallbacks/clipping/clamping reduce observability (Stage1 weight clipping, Stage2 regime clamping, calibration fallback). | Upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:31-36`, `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:49-55` | Calibration interpretability, regret metrics, entropy, coverage across A/B experiments | `src/capopm/corrections/stage1_behavioral.py`; `src/capopm/corrections/stage2_structural.py`; `src/capopm/metrics/calibration.py` |
| SD-06 | Structural prior surrogate deviates from paper Phase 1 model; fidelity gap not observable to audits. | Upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:12-19` | Any claim relying on Phase 1 prior fidelity (A1, B1, B3, B4, B5) | `src/capopm/structural_prior.py` vs paper Phase 1 specification |

## D) Stage A Closure Decision
- **Stage A closed under §2.2 (A1, A3, B1–B5): YES.** All scoped experiments have Stage A reports with documented failed/indeterminate criteria, causal traces through audit_contracts → audit.py → runner.py → experiment → upstream (or explicit paper semantic mismatches), and classifications assigned per §2.2. Systemic defects (paper–audit overreach, grid aggregation semantics, small-n untestability, fallback/clipping observability, structural prior fidelity gap, B2 visibility) are enumerated and tracked. A2 is excluded by governance and does not block completeness. requirements.txt is present (authoritative specification input) and not a blocker. Stage B.0 governance/spec work may proceed; no code changes are implied here.
