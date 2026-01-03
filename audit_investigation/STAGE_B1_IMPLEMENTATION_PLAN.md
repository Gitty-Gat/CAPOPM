# Stage B.1 Implementation Plan (Upstream Correctness & Observability)

Authority order: CAPOPM.pdf > requirements.txt > claim_table.md. Stage B.0 artifacts and Stage A reports are binding. No audit weakening is permitted; all changes are semantic corrections or observability upgrades only.

## 1) Scope Boundary (Modules in Scope and Rationale)
- **Audit plumbing & aggregation:** `src/capopm/experiments/audit.py`, `src/capopm/experiments/audit_contracts.py`, `src/capopm/experiments/paper_config.py`, `src/capopm/experiments/runner.py`. Why: SD-01 grid aggregation defect; AF-07 mean-only aggregation; AF-08 grid_points_observed; SD-02 over-claim removal where required by CLAIM_AUDIT_REMAP_TABLE.md; variance/CI surfacing; summary artifact updates.
- **Observability of fallbacks:** `src/capopm/corrections/stage1_behavioral.py`, `src/capopm/corrections/stage2_structural.py`, `src/capopm/metrics/calibration.py`, `src/capopm/metrics/scoring.py`, `src/capopm/experiments/projection_utils.py`, `src/capopm/structural_prior.py`. Why: AF-01–AF-06, AF-10; need logging/surfacing of clipping/clamping/eps; structural prior fidelity mode.
- **Experiment interfaces & summary writing:** `src/capopm/experiments/runner.py`, experiment scripts A1/A3/B1/B3/B4/B5; `run_paper_suite.py` if needed for grid aggregation data flow. Why: to propagate new observability fields and grid completeness to summary.json without altering claims.
- **B2 implementation discovery:** `src/capopm/experiments/b2_asymptotic_rate_check.py` (if present), related smoke test `smoke_test_b2_asymptotic_rate_check.py`, and any helper modules. Why: SD-04/R-06 visibility gap.
- **Documentation artifacts:** `audit_investigation/STRUCTURAL_PRIOR_FIDELITY_DECISION.md` (to be created), updates to governance tables if schema changes are required for observability fields.

“Upstream” includes any module affecting evidence→posterior→price→audit metrics or that can silently distort evaluation; exclusions: no new experiments, no threshold changes, no claim additions.

## 2) Systemic Defect to Fix Mapping (Each SD/AF Separate)
- **SD-01 (Grid aggregation defect):** Cause: `grid_points_observed` computed per scenario in `audit.py`/`runner.py` without cross-scenario aggregation. Affects A1/A3/B3/B4 grid_requirement. Code: `src/capopm/experiments/audit.py:342-359`; sweep handling in `runner.py:574-608` (Stage A cite).
- **SD-02 (Audit over-claims):** Cause: audit_contracts map paper claims to dominance/entropy/log-regret thresholds not supported by CAPOPM.pdf (per CLAIM_AUDIT_REMAP_TABLE). Affects A1/B1/B3/B4/B2/A3. Code: `src/capopm/experiments/audit_contracts.py` criteria definitions. Action limited to semantic corrections/removals of paper-forbidden metrics (no weakening).
- **SD-03 (Small-n untestable):** Cause: experiments run with n_runs below paper-ready; runner aggregates means only. Affects all A/B experiments. Code: `runner.py` aggregation; experiment configs. Stage B.1 action: surface variance/CI and low-n flags; no threshold relaxation.
- **SD-04 (B2 visibility gap):** Cause: Missing/opaque B2 implementation path. Code: locate `src/capopm/experiments/b2_asymptotic_rate_check.py` and smoke test; if absent, document.
- **SD-05 (Silent fallbacks/clipping/clamping):** Cause: Stage1 weight clipping, Stage2 clamping, calibration fallback, projection eps, log_score clamp not surfaced. Code: `stage1_behavioral.py`, `stage2_structural.py`, `metrics/calibration.py`, `metrics/scoring.py`, `experiments/projection_utils.py`.
- **SD-06 (Structural prior fidelity gap):** Cause: surrogate prior differs from Phase 1 spec and is silent. Code: `structural_prior.py`.
- **AF-01:** Stage1 clipping. Code: `stage1_behavioral.py:62-115`.
- **AF-02:** Stage2 clamping. Code: `stage2_structural.py:155-183`.
- **AF-03:** Calibration fallback. Code: `metrics/calibration.py:124-145`.
- **AF-04:** Projection eps clamp. Code: `experiments/projection_utils.py:6-46`.
- **AF-05:** Log_score clamp. Code: `metrics/scoring.py:10-26`.
- **AF-06:** Structural prior surrogate. Code: `structural_prior.py:18-70`.
- **AF-07:** Mean-only aggregation. Code: `runner.py:374-412`.
- **AF-08:** Per-scenario grid counting. Code: `audit.py:342-359` and runner sweep metadata.
- **AF-09:** conditional_on_violation auto-pass. Code: `audit.py:332-418`.
- **AF-10:** Alpha/beta positivity guard. Code: `likelihood.py:12-20` (observability only; no change unless logging).

## 3) Planned Change List (Ranked with Risks)
- **B1-CHG-01 (Grid aggregation fix)**  
  - Addresses: SD-01, AF-08, Risk R-02.  
  - Changes: Add cross-scenario grid aggregation in `runner.py` (sweep tracking) and `audit.py` (use aggregated grid_points_observed); write observed/missing grid keys to summary.json.  
  - Paper fidelity: Enables proper evaluation of grid-required claims without altering thresholds (alignment to Proposition 9, Theorem 15 grid expectations).  
  - Risks: Reproducibility (schema change), potential miscount if manifest wrong.  
  - Tests: All smoke tests A1/A3/B3/B4 plus manual check of summary.json grid fields; audit still fails if grid incomplete.  
  - Falsification: Grid counts mismatch actual scenario set; audits mis-evaluate requires_grid.

- **B1-CHG-02 (Observability of fallbacks/clamps)**  
  - Addresses: SD-05, AF-01–AF-05, AF-09, AF-10 (logging only), Risk R-04.  
  - Changes: Add deterministic logging and summary fields for Stage1 clipping counts/min/max, Stage2 clamping events, calibration fallback status, projection eps adjustments, log_score clamp activations, conditional_on_violation skip flag, alpha/beta guard trips.  
  - Paper fidelity: Observability only; no math changes.  
  - Risks: Schema expansion; potential performance if over-logged.  
  - Tests: Smoke tests A1/A3/B1/B3/B4/B5; verify new summary fields populated; ensure determinism.  
  - Falsification: New fields missing or inconsistent across identical reruns; behavior of metrics changes (should not).

- **B1-CHG-03 (Variance/CI surfacing)**  
  - Addresses: SD-03, AF-07, Risk R-03, R-07.  
  - Changes: In `runner.py`, store per-metric variance and CI/standard error (bootstrap or analytic where feasible) alongside means in aggregated metrics; flag low-n. No audit thresholds added.  
  - Paper fidelity: Truthful finite-sample reporting; no claim change.  
  - Risks: Added compute cost; schema change; potential CI methodology debates.  
  - Tests: Smoke tests; check summary.json for variance/CI; consistency across reruns.  
  - Falsification: Mean values change absent seed change; CI missing or nonsensical.

- **B1-CHG-04 (Audit contract semantic corrections)**  
  - Addresses: SD-02, Risk R-01.  
  - Changes: Remove/disable paper-forbidden metrics (dominance/entropy=0/log-regret under no-attack/slope sign) per CLAIM_AUDIT_REMAP_TABLE.md; retain strictness for admissible proxies.  
  - Paper fidelity: Aligns audits to CAPOPM.pdf; not a weakening—over-claims are semantic errors.  
  - Risks: Perceived regression in pass/fail counts; compatibility with claim_table expectations.  
  - Tests: Run smoke tests; ensure criteria now reflect admissible metrics; audit still evaluates where admissible.  
  - Falsification: Any removed criterion was actually paper-backed; higher authority (CAPOPM.pdf) would contradict removal.

- **B1-CHG-05 (Structural prior fidelity decision & logging)**  
  - Addresses: SD-06, AF-06, Risk R-05.  
  - Changes: Choose surrogate-accepted mode with explicit flag (e.g., `structural_prior_mode = "surrogate_v1"`) logged in summary; add invariant checks; create `STRUCTURAL_PRIOR_FIDELITY_DECISION.md` documenting limits and gating claims dependent on Phase 1 fidelity. No math invention.  
  - Paper fidelity: Explicitly gates claims relying on Phase 1; transparency on surrogate use.  
  - Risks: Schema change; may gate claims as UNTESTABLE.  
  - Tests: Smoke tests; check summary for mode flag and invariants; ensure no behavior change except logging/gating.  
  - Falsification: Mode flag absent; invariant failures not surfaced; claims not gated when mode ≠ paper-faithful.

- **B1-CHG-06 (B2 visibility instrumentation)**  
  - Addresses: SD-04, Risk R-06.  
  - Changes: Locate B2 experiment and smoke test; add explicit logging of rate computation (inputs/outputs) to summary; if absent, document in summary and audit_investigation. No semantic change unless code is missing/incorrectly referenced.  
  - Paper fidelity: Observability only; enables audit trace for Theorem 7 proxy.  
  - Risks: Schema change; risk of altering behavior if not careful.  
  - Tests: B2 smoke test (if present); verify summary contains rate computation details.  
  - Falsification: Behavior of rates changes; logging absent; B2 still opaque.

## 4) No-Audit-Weakening Guarantee
- Grid fixes (B1-CHG-01) alter only aggregation semantics to reflect actual sweeps; thresholds unchanged. Requires_grid remains strict but now evaluable.
- Observability logging (B1-CHG-02, B1-CHG-03) adds fields; no thresholds or metric comparisons are relaxed.
- Audit contract corrections (B1-CHG-04) remove paper-forbidden criteria per CLAIM_AUDIT_REMAP_TABLE.md; this is semantic alignment, not weakening. Admissible criteria remain strict.
- Structural prior mode flag (B1-CHG-05) gates claims reliant on Phase 1 fidelity; does not soften admissible criteria.
- B2 visibility (B1-CHG-06) is instrumentation; no thresholds changed.

All changes will be annotated in code with plan IDs (B1-CHG-xx). If any ambiguity risks paper-semantic drift, implementation will halt for clarification.
