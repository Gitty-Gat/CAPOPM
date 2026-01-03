
---

# **AGENTS_STAGE_B.md**

**CAPOPM — Stage B Governance, Specification, and Execution Contract**  
**Status:** Canonical (supersedes all prior Stage-B guidance)  
**Applies to:** All Codex activity once Stage A is formally closed  
**Audience:** Senior-level quantitative developer (Codex), project supervisor (human)  

---

## 0. Authority Hierarchy (Non-Negotiable)

All reasoning, implementation, and validation MUST respect the following precedence order:

1. **CAPOPM.pdf** (canonical mathematical authority; treat as the canonical truth; no changes to core math without direct citation and explanation; permission required for updates)
2. **requirements.txt** (extracted/summarized formal spec including alignment rubrics, claim inventories, proxy policies, testability doctrines, and risk taxonomies; subordinate to paper)
3. **claim_table.md** (cross-referenced with PDF theorems/propositions; prioritizes validated statuses like Phase 4 consistency)
4. All code, audits, experiments, summaries, smoke tests, and reports

If any conflict exists:

* **Higher authority overrides lower authority**
* The conflict MUST be documented explicitly
* Codex MUST NOT “resolve” conflicts by altering math, changing assumptions, filling in parts with hallucinated ideas, or weakening audits
* Ask for clarification prior to proceeding on any ambiguities or updates

---

## 1. Purpose of Stage B

Stage B exists to:

* Act as a "face-lift" to improve rigor in core math implementation (e.g., market setup, traders, priors, hybrid prior, likelihood, two-stage corrections for behavioral biases like long-shot/herding, nonlinear structural distortions via regime mixtures, sequential updating/time dynamics, posterior predictive derivative pricing)
* Translate **paper-faithful mathematics** into **paper-faithful code** without new math, changed assumptions, or hallucinated ideas
* Implement 118 pages of math programmatically, focusing on Phases 1-7 with theoretical guarantees from Phase 8
* Correct **semantic mismatches** identified in Stage A, using alignment rubrics to avoid overinterpretation (e.g., metrics derive from theorem implications like monotonicity in tail probabilities under fractional volatility from Theorem 1)
* Improve **observability, diagnosability, and testability** while preserving paper invariants (e.g., tail monotonicity in structural prior)
* Prepare the system for **paper-grade reruns** and **real-data ingestion**
* Preserve **academic rigor**, **finite-sample discipline**, and **audit integrity**; no claims of empirical superiority or dominance

Stage B is **not**:

* A phase for coming up with new math or extensions (extensions modular and disabled by default; paper claims cannot depend on them unless flagged as novel)
* A performance-tuning or validation-gaming phase
* A “make tests pass” phase; prioritize correctness over pass rates
* A production hardening phase

---

## 2. Preconditions: Stage A Closure Checklist (Hard Gate)

**Stage B MUST NOT begin** unless ALL items below are satisfied.

### 2.1 Required Artifacts (Presence Check)

The following MUST exist:

* Stage A reports for **all experiments** (e.g., A1–A3, B1–B5)
* `UPSTREAM_PIPELINE_AUDIT.md`
* `borderline_atlas.md`
* `claim_table.md`

### 2.2 Completeness Conditions

* Every failed or indeterminate audit criterion in Stage A:
  * Is documented
  * Is traced causally
  * Is classified (paper mismatch, finite-sample untestable, audit overreach, etc.)
* All **systemic defects** are enumerated as tracked items, including (at minimum):
  * Paper ↔ audit semantic overreach
  * Grid aggregation semantics
  * Fallback / clipping invisibility
  * Structural prior fidelity gap

If ANY item is missing → **STOP**.

---

## 3. Global Prohibitions (Apply to All Stage B Sub-Stages)

Codex MUST NOT:

* Change or reinterpret mathematical claims in CAPOPM.pdf (e.g., no overclaims like dominance without explicit theorem support)
* Weaken audit thresholds, gates, or criteria (e.g., no exploratory metrics labeled as aligned)
* Introduce dominance, optimality, or convergence claims not explicitly stated in the paper (e.g., no unsubstantiated dominance claims per proxy policy)
* Bucket multiple failures into generic fixes
* Modify historical artifacts under `results/`
* Claim empirical validation unless `paper_ready == True` under required run conditions (validation declarations prohibited)
* Implement proprietary logic inside the paper-core pipeline (extensions off by default)
* Overinterpret theorems (use alignment rubric: aligned for logical implications, proxy for directional trends, exploratory for unsupported, mismatch for stronger claims)
* Treat borderline notes as claims (inventory from claim_table.md prioritizes validated statuses)
* Use inadmissible proxies (e.g., regret dominance forbidden without explicit statements)
* Ignore minimum regimes (e.g., asymptotic theorems require increasing n-grids with CIs; untestable cases labeled "paper-backed but untestable in synthetics")
* Assume ground truth for regimes in real markets (use proxies like settlement outcomes; synthetics use p_true and regime labels)
* Focus on non-binary derivatives or non-parimutuel venues (target spec: binary derivatives in parimutuel with binary payoffs at T, evaluated via posterior means)

Violation of any prohibition invalidates the output.

---

## 4. Stage B Structure (Gated, Sequential)

Stage B is divided into **explicit sub-stages**. Progression is strictly sequential.

---

## **Stage B.0 — Governance & Specification (NO CODE)**

### Objective

Establish an unambiguous, enforceable specification so that **all later code changes are mathematically justified and auditable**.

### Mandatory Deliverables

#### 4.1 Formalized Goals Document

Must define:

* Target problem class (derivative pricing in behavioral parimutuel markets; binary options/YES/NO contracts; venue: prediction platforms; horizon: maturity T > 0; payoff: 1 if event occurs, 0 otherwise; evaluation: posterior mean ˆp; why CAPOPM fits: integrates crowd signals with structural/ML priors for belief extraction in distorted markets)
* Admissible markets (asset class: derivatives; inadmissible: continuous vs. parimutuel mismatches)
* Inadmissible markets (explicit exclusions per Assumption A2 price-taking, etc.)
* What CAPOPM is *not* claiming to solve (no empirical dominance per abstract; limitations in Section 1.6)

#### 4.2 Canonical Claim Index

For **every theorem (1-36), proposition (1-19), lemma (1-17), assumption (1-11), definition (1-10), and remark (1-32)** in CAPOPM.pdf:

* Identify with ID, reference, observables (e.g., convergence time), and stage (A/B)
* Classify as: paper-faithful & testable; paper-faithful but untestable (synthetic); exploratory only; audit overreach
* Cross-reference to **THEORY_APPENDIX.md** for full details including claims in plain text, metrics to test, definition of success, pipeline implementation, allowed/forbidden metrics, min sample regimes, interconnections, and dependencies.

See **THEORY_APPENDIX.md** for comprehensive tabular mappings, proxies, and testability status.

#### 4.3 Executable Specification Document

A phase-by-phase mapping:

```
Paper Phase → Module(s) → Inputs → Outputs → Invariants → Audit Touchpoints
```

Incorporate structural prior fidelity criterion (surrogates acceptable if matching outputs like tail monotonicity; full fractional Heston out-of-scope; acceptance tests: tail probability monotonicity, positivity, hybrid fusion robustness).

#### 4.4 Assumption Registry

Structured registry covering all 11 assumptions:

* ID, description, location, activation, impact, audit (e.g., Assumption A1: risk neutrality; impacts equilibrium strategies; audit: sensitivity to non-linear utils)

No undocumented fallback may remain.

#### 4.5 Risk Register & Priority Queue

Risks categorized by type (epistemic from misspecification, statistical from dependence violations, numerical from approximation errors, reproducibility from seed variance, scope from extension beyond parimutuel); scored by severity-likelihood; correctness prioritized over pass rates.

### Acceptance Criteria

Stage B.0 is complete **only if**:

* All deliverables exist
* No code was modified
* Unresolved ambiguities are explicitly listed

---

## **Stage B.1 — Upstream Correctness & Observability (LIMITED CODE)**

### Objective

Make the **core pipeline paper-faithful, observable, and testable**.

### Allowed Targets

* `structural_prior.py` (preserve Phase 1 invariants like tail monotonicity)
* `hybrid_prior.py`
* `ml_prior.py`
* `market_simulator.py`
* `trader_model.py` (any modules defining information sets, herding, whales/adversaries)
* `likelihood.py`
* `posterior.py`
* Trade-to-evidence mapping modules (weighted bets as evidence units; handle adverse selection)
* `corrections/stage1_behavioral.py` (long-shot/herding corrections)
* `correction2/stage2_structural.py` (nonlinear distortions via regime mixtures)
* `pricing.py` (posterior predictive derivative pricing)
* Projection utilities/modules (arbitrage-free simplex)
* `metrics/scoring.py`
* `metrics/calibration.py` (tied to theorem properties, e.g., entropy trends for robustness)
* Coverage interval logic
* `runner.py` and `audit.py` (fix aggregation semantics, diagnostics surfacing; no relaxing criteria)
* Diagnostics/logging in likelihood, posterior, corrections, projection
* Grid aggregation logic (per experiment family; min_grid=2; missingness fails audits)

### Mandatory Decisions

#### 4.6 Structural Prior Fidelity Decision

Choose and document:

* **Surrogate accepted** → with explicit limitations on testable claims (must preserve positivity, integrability, tail monotonicity)
* **Phase-1-faithful implementation** → minimal, preserving paper semantics; no over-engineering

### Mandatory Actions

* Surface all fallback activations into `summary.json` (include fallback counters, bin mode switches as required fields)
* Enforce invariants with explicit errors (unless paper allows otherwise)
* Correct grid aggregation for paper-defined sweeps
* Add unit tests for invariants, smoke tests for schemas/determinism, minimal paper suite (no validation declarations)
* NO new claims added

### Acceptance Criteria

* No change to core math claims
* New diagnostics visible to audits
* Tests pass
* Risks updated

---

## **Stage B.2 — Paper-Grade Reruns (NO INTERPRETATION)**

### Objective

Execute the **paper-defined experiment suite** under paper-ready conditions.

### Requirements

* `run_paper_suite.py`
* Required grid density (e.g., increasing n-grid 50-1000 for asymptotics)
* Required `n_runs` per cell
* Bootstrap / CI logic where specified (e.g., Berry-Esseen for normality approximations)

### Prohibitions

* No interpretation of results
* No narrative framing
* No “near pass” arguments

### Acceptance Criteria

Only one condition matters:

```
audit.overall_pass == True
AND
paper_ready == True
AND
criteria_semantics_mismatch == False
```

---

## **Stage B.3 — Tier C/D Experiments (EXPANSION ONLY AFTER CORRECTNESS)**

* Only after Stage B.2
* Must follow the same audit discipline
* Cannot introduce new claims without paper backing

---

## **Stage B.4 — Databento Integration (ADAPTER LAYER ONLY)**

### Rules

* Databento ingestion MUST be isolated
* Produce canonical event streams
* Evidence builder must be versioned and auditable
* No changes to Bayesian core allowed here

---

## **Stage B.5 — Proprietary Extensions (OPTIONAL)**

### Governance

* Must live in a separate namespace
* Disabled by default
* Cannot satisfy paper claims
* Must be labeled as new contributions

---

## 5. Documentation Standards (Hard)

All Stage B outputs MUST:

* Include file + line references
* Separate **paper-backed facts** from **engineering choices**
* Include a risks section
* Include a “what would falsify this” section

Minimal responses are forbidden.

---

## 6. Failure Protocol

If Codex encounters:

* Missing information
* Conflicting authorities
* Ambiguous paper language
* Untestable claims

It MUST:

1. STOP
2. State the ambiguity
3. Cite sources
4. Request clarification

Proceeding by assumption is forbidden.

---

## 7. Transition to THEORY_APPENDIX.md

AGENTS_STAGE_B.md contains:

* Canonical claim index (IDs with brief refs)

Full details (tabular mappings, proxies, metrics, etc.) are in **THEORY_APPENDIX.md**.

This file will be created **after** AGENTS_STAGE_B.md is accepted.

---

## 8. Final Principle

**Passing an audit is never evidence of correctness. Correctness is the precondition for audits to matter.**

Stage B exists to enforce that principle, with empathy to complexities like finite-sample noise and behavioral biases.

---

**END OF CONTRACT**



