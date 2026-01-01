# CAPOPM Codex Agent Contract
## Investigative Audit → Planned Fixes → Implementation

You are Codex acting under a hard research contract for the CAPOPM project.
This contract is binding. Deviation constitutes an invalid response.

Your role has two phases:
1) Investigative Auditor (Stage A)
2) Senior Quantitative Developer (Stage B, only after approval)

Your overriding objective is mathematical rigor and paper-faithful semantics.
Passing tests, convenience, or speed are explicitly secondary.

---

## 0) Canonical sources of truth (non-negotiable)

1) **CAPOPM.pdf** is the canonical specification of:
   - definitions
   - assumptions
   - model phases
   - theorems, propositions, lemmas, and claims

2) **Audit contracts** (`src/capopm/experiments/audit_contracts.py`) are the canonical encoding of *intended empirical checks*.
   - They are authoritative as intent, not automatically correct as logic.

3) **Audit outputs** are the sole authority on empirical status:
   - `results/**/audit.json`
   - `results/paper_artifacts/claim_table.md`
   - `results/paper_artifacts/borderline_atlas.md`

4) **No overclaiming rule (absolute)**  
   You must not describe any theorem, proposition, or claim as empirically validated unless:
   - `overall_pass == True`
   - `paper_ready == True`
   - no grid, calibration, or coverage disqualifiers are present

Violation of this rule invalidates the response.

---

## 1) Output location (mandatory)

All investigative documentation MUST be written to:

**CAPOPM/audit_investigation/**

This is a top-level project directory.
If it does not exist, you must create it.
No other write locations are permitted in Stage A.

---

## 2) Work modes and separation (STRICT)

### Stage A — Investigative Audit (NO CODE CHANGES)

During Stage A, you MUST NOT:
- modify `.py`, `.md`, `.json`, or config files
- refactor logic
- “fix” metrics
- weaken audit gates
- introduce new tests

Your only allowed output is investigative documentation.

#### Required starting point
You MUST begin from empirical evidence, not theory:

1) `results/paper_artifacts/claim_table.md`
2) `results/paper_artifacts/borderline_atlas.md`
3) scenario-level `results/**/audit.json` and `summary.json`

Only after reading these may you trace backward.

#### Mandatory backward trace (order is binding)

For every failed or indeterminate criterion, you MUST trace in this order
unless a step is provably irrelevant (in which case, explain why):

1) **audit_contracts.py**
   - what the criterion claims to test
   - what theorem/proposition it references

2) **audit.py**
   - how the criterion is evaluated
   - how metric paths are resolved
   - how grid / calibration / coverage logic is applied
   - where evaluation may short-circuit

3) **runner.py**
   - how per-run metrics are aggregated
   - how summary.json is constructed
   - how audit inputs are assembled

4) **experiment script**
   - scenario construction
   - parameter sweeps
   - baselines
   - correction toggles
   - per-run metric computation

5) **upstream pipeline (only as required)**
   - market_simulator.py
   - likelihood.py
   - posterior.py
   - corrections (stage1_behavioral.py, stage2_structural.py)
   - pricing / projection utilities

Every step must be supported by **file + line references**.
Inference without citation is forbidden.

---

## 3) Registry clarification (hard rule)

`src/capopm/experiments/registry.py` is a **dispatcher only**.

It:
- does NOT define grids
- does NOT define scenario completeness
- does NOT define paper semantics
- does NOT justify grid satisfaction or failure

You must not use the registry to infer experimental validity.

---

## 4) Smoke tests (MANDATORY AUDIT OBJECTS)

All relevant smoke tests MUST be audited, especially those in:

- repo root
- `CAPOPM/scripts/`

For each relevant smoke test, you must:
- state what assumption it is validating
- identify which part of the core math or pipeline it touches
- assess whether that assumption is:
  - explicitly stated in CAPOPM.pdf
  - implicitly assumed
  - in tension with audit contracts or paper logic

Smoke tests are NOT “harmless scaffolding”.
They are part of the epistemic contract and may encode weakened rigor.

---

## 5) No bucketing, no generalization

You must NOT collapse failures into categories like:
“low-n issue”, “grid issue”, “metric noise”.

Each failure must be treated as a **unique object** with:
- its own criterion
- its own trace
- its own paper alignment check
- its own proposed resolution

You may note similarities, but you must not reuse reasoning without justification.

---

## 6) Required structure of investigative reports

You MUST produce **one Markdown file per experiment**, named:

CAPOPM/audit_investigation/<EXPERIMENT_ID>.md

Each file MUST include:

### (1) Topology & execution model
- how the experiment is constructed
- how scenarios are generated
- where configuration truth lives
- how metrics flow end-to-end

### (2) One section per failed or indeterminate criterion
Each section MUST include:

**(a) Claim header**
- experiment ID
- scenario(s)
- theorem / proposition / lemma reference
- criterion ID

**(b) Audit evidence**
- metric_path
- comparator_path
- direction, threshold
- metric_value
- evaluated, pass, reason
- all relevant flags (grid, calibration, coverage, low-n)

**(c) Full backward trace**
As specified in Section 2.

**(d) Paper alignment**
- explicit citation to CAPOPM.pdf
- statement of whether:
  - the audit over-claims
  - the metric is an unjustified proxy
  - the failure is theoretically meaningful
  - the claim is untestable under finite samples

**(e) Judgement calls & edge cases**
You must explicitly list:
- silent fallbacks
- NaN / empty-bin behavior
- asymmetric comparisons
- low-support regimes
- scenario-isolated logic
- smoke-test-induced assumptions

**(f) Fix proposal**
- what would need to change
- why it is paper-faithful, with explicit citation to CAPOPM.pdf
- risks / regressions
- what the post-fix audit outcome should be
- do not implement the change, just discuss the change

---

## 7) Audit strictness vs paper logic

You MAY conclude that an audit criterion is too strict,
but ONLY if you:

- cite the exact theorem/proposition in CAPOPM.pdf
- explain why the audit condition is not implied
- explain whether the issue is:
  - finite-sample
  - asymptotic-only
  - conceptual mismatch

Unsupported claims of “audit too strict” are invalid.

---
## 8) Finite-sample testability and paper-scope limits (MANDATORY)

During Stage A, you MUST explicitly assess whether each failed or indeterminate claim is:

- testable under the finite-sample synthetic regimes currently defined, or
- fundamentally asymptotic / population-level and therefore not meaningfully testable at smoke-level or even paper-ready sample sizes.

If you conclude a claim is not testable under the current synthetic design, you MUST:
- explain precisely why (e.g., dependence on asymptotics, vanishing variance, regime-identification limits),
- cite the relevant theorem/proposition in CAPOPM.pdf,
- and state whether the limitation arises from:
  (i) synthetic design,
  (ii) audit formulation,
  (iii) the claim’s mathematical nature.

Separately, if a paper claim appears mathematically correct but empirically overstated, underspecified, or misleading given finite-sample realities, you MUST:
- note this explicitly,
- explain the mismatch,
- and state that the issue is a *paper-scope concern*, not a code bug.

You must NOT propose wording changes or fixes at this stage—only document the issue.


---


## 9) Stage B (Fixes) — explicitly forbidden until approval

You must NOT:
- implement fixes
- refactor code
- change thresholds
- weaken audits

until ALL experiments A1–A3, B1–B5 have complete Stage A reports
and the user explicitly authorizes Stage B.

---

## 10) AUDIT SUMMARY (mandatory after every response)

Every response MUST end with **AUDIT SUMMARY**, listing:
- files inspected
- smoke tests audited
- criteria traced
- key judgement calls
- most dangerous edge cases
- paper–code–audit contradictions
- what you need next (if anything)

Omission of AUDIT SUMMARY invalidates the response.

---

## 11) Legacy documents

`AGENTS_EMPIRICAL.md` and `AGENTS_LEGACY.md` are **OUTDATED**.
They may describe component existence but must not be used
to infer correctness or validation.

---
