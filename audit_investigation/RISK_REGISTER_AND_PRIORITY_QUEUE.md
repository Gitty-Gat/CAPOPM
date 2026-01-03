# Risk Register and Priority Queue (Stage B.0)

Authority: CAPOPM.pdf > requirements.txt > claim_table.md. Risks classified by category with severity × likelihood rubric. Priorities focus on correctness and paper alignment, not audit pass rates.

## Severity × Likelihood Rubric
- Severity: 1 (negligible), 2 (minor), 3 (material), 4 (critical), 5 (existential to claim validity).
- Likelihood: 1 (rare), 2 (unlikely), 3 (possible), 4 (likely), 5 (certain under current configs).
- Risk Score = Severity × Likelihood (max 25).

## Risk Register
| ID | Risk | Category | Evidence/Citation | Severity | Likelihood | Score | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-01 | Audit contracts over-claim paper theorems (dominance, entropy=0, slope sign) | Epistemic | SD-02 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:37-40`; A1/A3/B1/B2/B3/B4 reports | 4 | 4 | 16 | Misaligned claims invalidate empirical interpretation. |
| R-02 | Grid aggregation defect (per-scenario grid_points_observed=1) | Statistical / Reproducibility | SD-01 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:34-36`; A1/A3/B3/B4 grid indeterminates | 3 | 5 | 15 | Causes systematic indeterminacy; prevents reproducible evaluation. |
| R-03 | Small-n regimes below paper-ready thresholds | Statistical | SD-03 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:41-43`; A1/A3/B1/B3/B4/B2 | 4 | 4 | 16 | Finite-sample noise makes claims untestable and can flip signs. |
| R-04 | Silent fallbacks/clipping/clamping (Stage1, Stage2, calibration) | Observability / Numerical | SD-05 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:44-47`; upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:31-36,49-55` | 4 | 4 | 16 | Hidden alterations to data/metrics compromise diagnostics. |
| R-05 | Structural prior surrogate fidelity gap vs CAPOPM.pdf Phase 1 | Epistemic / Scope | SD-06 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:48-50`; upstream audit `audit_investigation/UPSTREAM_PIPELINE_AUDIT.md:12-19` | 5 | 3 | 15 | Core math divergence risks invalidating paper-backed claims. |
| R-06 | Missing visibility into B2 rate computation implementation | Reproducibility | SD-04 `audit_investigation/STAGE_A_CLOSURE_CHECKLIST.md:41-43`; B2 report `audit_investigation/B2.ASYMPTOTIC_RATE_CHECK.md:22` | 3 | 3 | 9 | Without code trace, asymptotic rate claims cannot be validated or fixed. |
| R-07 | Mean-only aggregation masking variance/CI | Statistical | AF-07 `audit_investigation/ASSUMPTION_FALLBACK_REGISTRY.md` | 3 | 4 | 12 | Overstates confidence, particularly in small-n settings. |
| R-08 | Projection epsilon/clamp biases distance metrics | Numerical | AF-04 `audit_investigation/ASSUMPTION_FALLBACK_REGISTRY.md` | 2 | 3 | 6 | Could understate violation severity in B5; moderate impact. |

## Priority Queue for Stage B.1
Ordered by risk score and dependency on paper correctness. “Making audits pass” is excluded because alignment to CAPOPM.pdf is prerequisite; passing misaligned audits would entrench incorrect claims.

1) **R-01: Audit contract over-claims** (Score 16) — prerequisite to align audits to paper; prevents false validation.  
2) **R-03: Small-n regimes** (Score 16) — must enforce paper-ready run counts/CI before interpreting metrics.  
3) **R-04: Silent fallbacks/clamping** (Score 16) — surface diagnostics to avoid hidden behavior.  
4) **R-05: Structural prior fidelity gap** (Score 15) — decide surrogate acceptance vs Phase-1-faithful implementation with citations.  
5) **R-02: Grid aggregation defect** (Score 15) — implement cross-scenario aggregation to evaluate grid-required claims properly.  
6) **R-06: B2 visibility gap** (Score 9) — locate/verify rate computation before altering audits.  
7) **R-07: Mean-only aggregation** (Score 12) — add variance/CI reporting to support finite-sample interpretation.  
8) **R-08: Projection epsilon bias** (Score 6) — quantify/log projection adjustments.

## Why “Making Audits Pass” Is Not a Priority Criterion
Passing current audits could validate over-claimed or misaligned criteria (R-01) and would ignore small-n invalidity (R-03). Priority is given to paper-faithful correctness, observability, and statistical adequacy; audit outcomes are meaningful only after alignment to CAPOPM.pdf and requirements.txt.
