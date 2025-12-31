# AGENTS_EMPIRICAL.md  
**CAPOPM Empirical Validation Roadmap (Authoritative Contract)**

This document defines the **canonical empirical validation program** for CAPOPM.  
It is written to be *machine-legible* and *human-auditable*, and should be treated as a **binding contract** for future development, experiments, and evaluation.

> **Status note:** Phases 1–7 of CAPOPM are implemented and validated.  
> This roadmap governs all remaining empirical work.

---

## 1. Why the Tier Structure Exists

The empirical program is **tiered** to mirror the logical structure of the CAPOPM paper and to prevent premature claims.

### Tier meanings

- **Tier A — Mechanism validation**  
  *Question:* “Do the microstructure mechanisms actually work as claimed?”  
  Focuses on *local behavior*: trader interactions, distortions, and corrections.

- **Tier B — Theoretical consequence validation**  
  *Question:* “Do the paper’s theorems and propositions manifest empirically?”  
  Focuses on *global consequences*: regret, consistency, asymptotics, constraints.

- **Tier C — Stress & falsification**  
  *Question:* “Where does CAPOPM fail, and how does it fail?”  
  Focuses on *boundary conditions* and adversarial regimes.

- **Tier D — Reproducibility & publication artifacts**  
  *Question:* “Can an external researcher reproduce every claim?”  
  Focuses on *infrastructure, figures, tables, and failure documentation*.

> **Rule:**  
> An experiment in Tier B may not be claimed until all prerequisite Tier A experiments pass.  
> Tier C experiments are required before publication-level claims.  
> Tier D artifacts are mandatory for submission.

---

## 2. Standardized Experiment Contract

Every experiment **must** produce the following outputs:

### Required Outputs (Standardized)
- `metrics_aggregated.csv`
- `reliability_<model>.csv` (one per model)
- `tests.csv`
- `summary.json`

### Required Metadata Columns
- `scenario_name`
- `experiment_id`
- `tier`
- `model`
- `seed`

### Required Metric Families
Metrics must be chosen based on the **theoretical claim being tested** (see per-experiment mapping below):

- Proper scoring rules:  
  - Brier score  
  - Log score (Theorem references in Phase 5)
- Calibration:  
  - ECE (full-sample)  
  - Reliability tables
- Bias & consistency:  
  - Posterior mean bias  
  - MAE vs \( p_{\text{true}} \)
- Uncertainty behavior:  
  - Posterior variance ratio  
  - Coverage (90%, 95%) wrt \( p_{\text{true}} \)
- Geometry / constraints (where relevant):  
  - Wasserstein distance (Beta space)  
  - Projection distance (simplex / arbitrage tests)

---

## 3. Tier A — Mechanism Validation (Microstructure Logic)

### A1. **INFO_EFFICIENCY_CURVES**
**Claim tested:**  
CAPOPM aggregates information efficiently as trader signal quality increases  
(Phase 4; Proposition 6; Lemma 3).

**Design:**  
Sweep informed share, signal quality, adversarial share.

**Metrics:**  
Learning curves, final Brier, convergence bias.

**Pass condition:**  
Monotonic improvement; CAPOPM dominates raw parimutuel.

---

### A2. **TIME_TO_CONVERGE**
**Claim tested:**  
CAPOPM stabilizes faster under higher liquidity  
(Implicit in Phase 4 consistency arguments).

**Design:**  
Sweep arrivals × steps × pool seeding.

**Metrics:**  
Time-to-ε error, posterior variance decay.

---

### A3. **STRATEGIC_TIMING_ATTACK**
**Claim tested:**  
Stage 1 + Stage 2 corrections reduce vulnerability to late manipulation  
(Theorem 12; Proposition 9).

**Design:**  
Adversarial traders concentrate trades late.

**Metrics:**  
Final bias, regret vs uncorrected.

---

### A4. **LONGSHOT_BIAS_SWEEP**
**Claim tested:**  
Behavioral correction removes long-shot bias  
(Phase 6.2; Lemma 8).

**Design:**  
Inject long-shot preference into trader behavior.

**Metrics:**  
Bias before/after Stage 1.

---

### A5. **HERDING_CASCADE_RESPONSE**
**Claim tested:**  
Herding correction dampens cascades without destroying information  
(Theorem 11).

**Design:**  
Correlated arrivals + streak following.

**Metrics:**  
Posterior variance inflation; ECE change.

---

### A6. **WHALE_DOMINANCE_MAP**
**Claim tested:**  
CAPOPM remains stable under concentrated liquidity  
(Proposition 10).

**Design:**  
Scale whale trade sizes and clustering.

**Metrics:**  
Error, regime diagnostics, stability flags.

---

### A7. **REGIME_SEPARATION_SANITY**
**Claim tested:**  
Stage 2 regime mixture recovers true distortion regimes  
(Theorem 15).

**Design:**  
Simulate known regimes.

**Metrics:**  
Regime weights, entropy, recovery accuracy.

---

## 4. Tier B — Theoretical Consequence Validation

### B1. **CORRECTION_NO_REGRET**
**Claim tested:**  
Corrections impose constraints without increasing regret  
(Theorem 14).

**Metrics:**  
Expected log-score regret distributions.

---

### B2. **ASYMPTOTIC_RATE_CHECK**
**Claim tested:**  
Consistency and variance decay  
(Theorem 7; asymptotic normality discussion).

**Metrics:**  
Bias → 0; variance ∝ 1/n.

---

### B3. **MISSPECIFICATION_REGRET_GRID**
**Claim tested:**  
CAPOPM minimizes regret under prior misspecification  
(Proposition 8).

**Metrics:**  
Regret surfaces across miscalibration grid.

---

### B4. **REGIME_POSTERIOR_CONCENTRATION**
**Claim tested:**  
Mixture posterior concentrates with strong evidence  
(Theorem 15).

**Metrics:**  
Entropy ↓; max regime weight ↑.

---

### B5. **ARBITRAGE_PROJECTION_IMPACT**
**Claim tested:**  
Projection minimally perturbs prices unless constraint violated  
(Theorem 13).

**Metrics:**  
Projection distance vs score improvement.

---

## 5. Tier C — Stress & Falsification

### C1. **ZERO_INFORMATION_MARKET**
Tests non-hallucination: posterior reverts to prior.

---

### C2. **ADVERSARY_MAJORITARIAN**
Finds failure boundary when adversarial volume dominates.

---

### C3. **NONSTATIONARY_PTRUE_DRIFT**
Tests robustness under model misspecification over time.

---

### C4. **REGIME_SWITCH_MIDWINDOW**
Tests regime-mixture lag and recovery.

---

### C5. **LIQUIDITY_DROPOUT**
Tests thin-market behavior and CI reliability.

---

### C6. **EXTREME_PRIOR_MISLEAD**
Tests recovery from severely wrong priors.

---

## 6. Tier D — Reproducibility & Publication Artifacts

### D1. **SCENARIO_REGISTRY_INDEX**
Single authoritative registry of all experiments.

### D2. **FIGURE_FACTORY**
Deterministic generation of all paper figures.

### D3. **TABLE_FACTORY**
Deterministic generation of all tables.

### D4. **REPRO_CHECKPOINTS**
One-command reproduction with hashes.

### D5. **FAILURE_ATLAS_REPORT**
Structured catalog of known failure modes.

---

## 7. Completion Tracking (Machine-Readable)

Each experiment must write to:
results/<scenario_name>/status.json
With fields:

{json}
{
    "experiment_id": "A1.INFO_EFFICIENCY_CURVES",
    "tier": "A",
    "completed": true,
    "passed": true,
    "date": "YYYY-MM-DD",
    "git_commit": "<hash>"
    "notes": ""
}
Codex MUST NOT mark Tier B/C experiments as valid unless all dependencies are marked passed.

---

## 8. Final Rule

CAPOPM is considered empirically complete only when:
    - "All Tier A and B experiments pass"
    - "Tier C failure modes are documented"
    - "Tier D artifacts exist and reproduce end-to-end"
    
---