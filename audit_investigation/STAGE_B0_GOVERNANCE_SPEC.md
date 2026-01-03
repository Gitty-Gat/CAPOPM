
---

# Stage B.0 Governance Spec — **Locked Addendum**

*(Surrogate-Only Structural Prior, Invariant-Level Acceptance)*

This addendum is **binding** and supplements the Stage B Governance Spec previously drafted. It resolves all remaining ambiguity around invariant domains, numerical tolerances, and surrogate admissibility.

---

## 1. Structural Surrogate Acceptance Domain (Binding Defaults)

These defaults define the **canonical domain** for all Stage B.1–B.2 invariant checks unless explicitly overridden and justified in writing.

### 1.1 Strike (Moneyness) Grid

* **Definition**:
  [
  K \in [0.5S_0,;1.5S_0]
  ]
* **Grid density**: 10–20 evenly spaced points in moneyness.
* **Purpose**:

  * Covers ITM / ATM / OTM regimes without entering extreme tails where numerical artifacts dominate.
  * Matches common Heston calibration practice.
* **Governance classification**:

  * **E-proxy domain** for tail-monotonicity and sensitivity diagnostics.
  * **Not** a claim about asymptotic or extreme-tail correctness.

> Any extension beyond this range must be explicitly labeled **Exploratory (X)** and cannot be used to support invariant acceptance.

---

### 1.2 Maturity

* **Default**:
  [
  T = 1 \text{ year}
  ]
* **Rationale**:

  * Long enough for stochastic volatility dynamics to manifest.
  * Short enough to avoid numerical instability or long-horizon extrapolation artifacts.
* **Classification**:

  * **Operational default**, not paper-mandated.
  * Changes in (T) do not re-interpret paper theorems unless explicitly stated.

---

### 1.3 Rates and Dividends

* **Defaults**:
  [
  r = 0,\quad q = 0
  ]
* **Justification**:

  * Simplifies baseline surrogate behavior.
  * Eliminates confounding drift effects when testing volatility-driven invariants.
* **Governance note**:
  Any nonzero (r) or (q) must be treated as a **domain extension**, not a paper-backed necessity.

---

## 2. Numerical Tolerance Policy (Operational, Non-Negotiable)

Numerical tolerance exists **only** to accommodate floating-point and approximation noise—not to relax mathematical requirements.

### 2.1 Invariant Classes and Tolerances

| Invariant Type                                                           | Tolerance             | Enforcement                | Notes                           |
| ------------------------------------------------------------------------ | --------------------- | -------------------------- | ------------------------------- |
| **Hard invariants** (positivity, integrability, bounded probabilities)   | **Absolute ≤ 1e-8**   | **Zero tolerance**         | Any violation is a hard failure |
| **Monotonicity / trend invariants** (e.g., tail probabilities vs strike) | **Relative ε = 1e-6** | Bounded deviations allowed | Violations must not accumulate  |
| **Exploratory diagnostics**                                              | ε ≤ 1e-4              | Informational only         | Never admissible for claims     |

### 2.2 Interpretation Rules

* A monotonicity check **passes** iff:

  * Violations are isolated,
  * Each violation magnitude ≤ ε,
  * Aggregate deviation is (O(\varepsilon^2)) or smaller.
* Repeated or structured violations ⇒ **surrogate rejection** (not theorem rejection).

> **Critical rule**:
> Passing these checks **does not validate any theorem**.
> Failing these checks **invalidates the surrogate or implementation**, not the paper.

---

## 3. Surrogate Governance (Final)

### 3.1 Allowed Structural Prior (Stages B.1–B.2)

* **Only permitted surrogate**: **Standard Heston**
* **Fractional Heston**: Explicitly **out-of-scope**
* **Neural / learned surrogates**:

  * **Forbidden** in B.1–B.2
  * Allowed **only** in **B.5**, isolated, disabled by default

### 3.2 Role of the Surrogate

The surrogate exists to:

* Preserve **qualitative invariants** (positivity, integrability, ordering).
* Enable **mechanism-level correctness checks** (likelihood mapping, Bayesian updates).
* Provide a **test harness**, not a replacement, for Phase 1 mathematics.

It **does not**:

* Approximate fractional long-memory,
* Enable testing of fractional asymptotics,
* Justify extrapolation to real-market dominance.

---

## 4. Falsification Criteria (Correctly Scoped)

The following **do not falsify CAPOPM.pdf**. They falsify **this surrogate-based implementation as an admissible proxy**.

### 4.1 Structural Proxy Failures

| Failure                                    | Interpretation                                 |
| ------------------------------------------ | ---------------------------------------------- |
| Tail probability non-monotonicity beyond ε | Surrogate numerics or parameterization invalid |
| Negative variance / invalid probabilities  | Immediate hard stop                            |
| Likelihood undefined / divergent           | Mapping or integrability failure               |

### 4.2 Explicit Non-Falsifications

* **Lack of regime posterior concentration**
  → Classified as:

  * finite-sample insufficiency,
  * regime unidentifiability,
  * surrogate-induced indistinguishability, or
  * observability mismatch
    **Never** as contradiction of Theorem 15.

---

## 5. Risk Register (Locked Entries)

### 5.1 Epistemic Risk — Fractional Dynamics Absent

* Severity: **High**
* Likelihood: **Medium**
* Mitigation:

  * Explicit quarantine of fractional-dependent theorems.
  * Invariant-only acceptance.

### 5.2 Numerical Risk — Tail Approximation

* Severity: **High**
* Likelihood: **Medium**
* Mitigation:

  * Strict monotonicity enforcement.
  * Grid-based diagnostics.
  * Hard rejection on repeated violations.

### 5.3 Dependence / Assumption Violations

* Severity: **High**
* Likelihood: **Unknown**
* Mitigation:

  * Explicit simulator-likelihood alignment checks.
  * No silent conditional-independence assumptions.

---

## 6. Stage B Execution Gate (Unlocked)

With this addendum:

✅ Structural-prior governance is frozen
✅ Invariant domains are explicit
✅ Numerical tolerances are enforceable
✅ Surrogate scope is unambiguous
✅ Claim inflation pathways are sealed

---

## Immediate Next Step (Authoritative)

### **Proceed to Stage B.1: Upstream Correctness & Observability Specification**

The next deliverable should **not** include code. It should:

1. Enumerate **every observable** (real or synthetic) used anywhere in the pipeline.
2. Map each observable to:

   * paper object,
   * simulator source,
   * admissible claim class (P / E / X).
3. Define **likelihood-factorization checks** and **dependency audits**.
4. Specify **trace logging** required to make failures attributable.

I