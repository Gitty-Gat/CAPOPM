# PRISM_CLAIM_BOUNDARIES.md
# PRISM Claim Boundaries (Binding)

**Purpose:** Prevent scope drift and hallucination. This document defines what PRISM *is*, what it *is not*, what is *validated*, and what must be labeled as *future work*.

---

## 1. What PRISM Is (Canonical)

PRISM is a **Bayesian workflow** that produces a **posterior distribution** over an event probability using:

1. **Granular belief data** from raw crowd order flow (L3 order book).
2. A defined method to extract **anchor beliefs** from that order flow.
3. An **ML-generated signal** that is treated as a probability estimate (or score) and then **calibrated**.
4. A **fusion** mechanism that combines anchor beliefs and calibrated ML information into a coherent probabilistic representation (preferably conjugate where feasible).
5. **Bias corrections** that adjust for known distortions (behavioral and/or structural), expressed as transparent transforms.
6. A final **posterior** over the event of interest.

PRISM’s output is interpretability-focused: posterior mean, uncertainty, and diagnostics.

---

## 2. What PRISM Is Not (Hard Exclusions)

PRISM is **not**:
- an option pricing model
- a risk-neutral pricing engine
- a replacement for structural asset pricing models
- a guarantee of market efficiency
- a claim of empirical dominance

If the paper mentions pricing-related uses, they must be framed as:
- downstream applications, or
- optional modules, or
- future work, clearly marked

---

## 3. Allowed Financial Framing

The manuscript may emphasize:
- financial engineering motivation
- market microstructure and order flow
- belief aggregation in financial settings
- decision-support under uncertainty

But must not imply:
- arbitrage-free pricing guarantees
- calibration to real option surfaces (unless you actually did it)
- trading profitability claims

---

## 4. Validation Boundaries

### 4.1 What can be claimed in the main paper
- The workflow definition and mathematical formulation.
- Internal consistency results (e.g., conjugacy preservation if applicable).
- Theoretical properties that are proven in-paper (under explicit assumptions).
- High-level synthetic validation summary (only if generated and reproducible).

### 4.2 What must be restricted to the supplement
- Full simulation suite details and stress tests
- extensive diagnostics
- large tables and plot grids

### 4.3 What must be labeled as future work
Any of the following, unless already completed and reproducible:
- real-market calibration studies
- full production-grade order book ingestion claims
- cross-market generalization claims
- claims of superior predictive accuracy vs established baselines

---

## 5. ML Component Boundary (Critical)

The ML component is allowed only under the following framing:

- ML provides a **signal** (raw probability estimate or score).
- That signal is **calibrated** using a transparent method.
- The paper must explain:
  - calibration dataset assumptions (if any)
  - calibration error metrics
  - uncertainty or limitations

**Forbidden:**
- “virtual sample size” as a definitive foundation unless validated
- claims that ML output is itself a prior
- black-box justification without calibration evidence

If ML-to-Beta parameter mapping is included:
- it must be described as a modeling choice
- with assumptions and limitations stated
- and with a conservative interpretation

---

## 6. Theorem/Proof Boundary

- A theorem may remain only if:
  1) its assumptions are still true under Beta(1,1) + calibration-based workflow, and
  2) it does not rely on deleted fractional Heston machinery.

If a theorem becomes invalid:
- remove it, or
- weaken it (more assumptions, narrower claim), or
- move to future work (if it is valuable but not currently defensible)

No theorem may remain that implicitly assumes PRISM is a pricing model.

---

## 7. Citation Boundary

- No new citations may be introduced unless they already exist in the project bibliography.
- If a citation is broken, unverifiable, or hallucinated:
  - remove it, and optionally leave a TODO comment.
- Never “patch” citations by inventing plausible references.

---

## 8. Diagram Boundary

Workflow diagrams (DAGs/box-arrows) must:
- reflect implemented or clearly specified steps
- avoid including speculative modules as if they exist
- label “optional” or “future” components clearly if shown
