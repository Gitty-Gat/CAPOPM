
---

# **STAGE B.1 — OBSERVABILITY CONTRACT + INVARIANT CHECKLIST**

**CAPOPM (Crowd-Adjusted Parimutuel Option Pricing Model)**

**Status:** AUTHORITATIVE / LOCKED
**Applies to:** All Stage B.1–B.5 work
**Precedence:** Subordinate only to CAPOPM.pdf and THEORY_APPENDIX.md
**Surrogate Regime:** Standard Heston only (fractional explicitly out-of-scope)

---

## 0. Purpose and Scope

This document defines, exhaustively and unambiguously:

1. **What is observable** in CAPOPM (real or synthetic)
2. **What is forbidden to observe or infer**
3. **How observables map to paper objects**
4. **Which invariants are admissible**
5. **How invariant failures are classified**
6. **What conclusions are categorically disallowed**

No Stage B execution, prompt, experiment, audit, metric, or visualization is valid unless it conforms to this contract.

This document **constrains agents, humans, and code equally**.

---

## 1. Observability Doctrine (Foundational)

### 1.1 Core Rule

> **No claim, invariant, diagnostic, or metric may depend on an object that is not explicitly declared observable in this contract.**

If an object is not listed below, it is **unobservable by definition**, regardless of whether code could technically access it.

---

### 1.2 Observable Classes

All observables fall into exactly one of the following classes.

| Class | Meaning                                                       | Claim Eligibility |
| ----- | ------------------------------------------------------------- | ----------------- |
| **R** | Real-market observable                                        | P / E             |
| **S** | Synthetic observable (simulator-generated, with ground truth) | E / X             |
| **L** | Latent but logged (simulator-only)                            | X only            |
| **F** | Forbidden / unobservable                                      | None              |

---

## 2. Canonical Observable Inventory

### 2.1 Trade-Level Observables (R, S)

Available from:

* `market_simulator.py`
* Kalshi data adapters (archive)
* `likelihood.py`

**Declared Observables**

| Name                  | Symbol          | Class | Description             |
| --------------------- | --------------- | ----- | ----------------------- |
| Trade timestamp       | (t_i)           | R / S | Time of trade           |
| Contract identifier   | (c_i)           | R / S | Binary event contract   |
| Trade side            | (y_i \in {0,1}) | R / S | Yes/No                  |
| Trade size            | (v_i)           | R / S | Number of contracts     |
| Execution price       | (p_i)           | R / S | Transaction price       |
| Market state snapshot | (M_{t_i})       | R / S | Best bid/ask, pool size |
| Event resolution      | (Z \in {0,1})   | R / S | Final outcome           |

---

### 2.2 Market-Level Aggregates (R, S)

Derived but admissible.

| Name                         | Symbol         | Class  |
| ---------------------------- | -------------- | ------ |
| Total volume                 | (V_t)          | R / S  |
| Yes/No pool sizes            | (Q_t^Y, Q_t^N) | R / S  |
| Implied raw parimutuel price | (\hat{p}_t)    | R / S  |
| Liquidity regime label       | (\ell_t)       | S only |

---

### 2.3 Structural Prior Outputs (S)

From `structural_prior.py` (surrogate-only).

| Name                             | Symbol                | Class | Notes            |
| -------------------------------- | --------------------- | ----- | ---------------- |
| Risk-neutral tail probability    | (q_{\text{sur}}(K,T)) | S     | Heston surrogate |
| Structural implied digital price | (D_{\text{sur}}(K,T)) | S     | Derived          |
| Structural parameter vector      | (\Theta)              | L     | Logged only      |

⚠️ **(\Theta) is latent and may not be used in any claim or invariant.**

---

### 2.4 ML Prior Outputs (S)

From `ml_prior.py`.

| Name           | Symbol          | Class |
| -------------- | --------------- | ----- |
| ML probability | (q_{\text{ML}}) | S     |
| Feature vector | (X_t)           | L     |

---

### 2.5 Posterior & Pricing Outputs (R, S)

From `posterior.py`, `pricing.py`.

| Name                    | Symbol        | Class |
| ----------------------- | ------------- | ----- |
| Posterior belief        | (q_t)         | R / S |
| CAPOPM price            | (\pi_t)       | R / S |
| Stage-1 corrected price | (\pi_t^{(1)}) | R / S |
| Stage-2 corrected price | (\pi_t^{(2)}) | R / S |

---

### 2.6 Regime & Correction Internals (L, S)

| Name             | Symbol     | Class |
| ---------------- | ---------- | ----- |
| Regime posterior | (\gamma_t) | S     |
| Whale indicator  | (W_t)      | L     |
| Herding weight   | (h_t)      | L     |

Latents **must never** appear in metrics or plots unless explicitly labeled *Synthetic-Only Diagnostic*.

---

## 3. Forbidden Objects (F)

The following are **explicitly forbidden**:

* True trader beliefs
* Counterfactual prices under alternative mechanisms
* Unobserved private signals
* “Efficiency gap” defined against unknown true probabilities
* Regret against an unobservable oracle
* Asymptotic limits evaluated at finite (n)

Any appearance of these invalidates the artifact.

---

## 4. Mapping to Paper Objects

| Paper Object                      | Observable Proxy  | Class          |
| --------------------------------- | ----------------- | -------------- |
| Binary payoff                     | (Z)               | R / S          |
| Parimutuel price                  | (\hat{p}_t)       | R / S          |
| Structural prior (q_{\text{str}}) | (q_{\text{sur}})  | S (proxy only) |
| Posterior belief                  | (q_t)             | R / S          |
| Regime mixture                    | (\gamma_t)        | S              |
| Arbitrage-free price              | Projection output | S              |

---

## 5. Invariant Checklist (Authoritative)

### 5.1 Structural Invariants (Hard)

| ID  | Invariant                | Inputs                | Tolerance  | Failure    |
| --- | ------------------------ | --------------------- | ---------- | ---------- |
| S-1 | Positivity               | (q_{\text{sur}})      | abs ≤1e-8  | Hard fail  |
| S-2 | Integrability            | likelihood terms      | none       | Hard fail  |
| S-3 | Tail monotonicity in (K) | (q_{\text{sur}}(K,T)) | rel ε=1e-6 | Proxy fail |

Domain fixed at:

* (K \in [0.5S_0,1.5S_0])
* (T=1), (r=0), (q=0)

---

### 5.2 Likelihood & Bayesian Invariants

| ID  | Invariant                | Meaning                        |                 |
| --- | ------------------------ | ------------------------------ | --------------- |
| B-1 | Likelihood normalization | (P(\text{data}                 | \theta)) finite |
| B-2 | Posterior normalization  | (\int q_t =1)                  |                 |
| B-3 | Conditioning correctness | Only uses declared observables |                 |

Failures ⇒ **implementation error**, not paper falsification.

---

### 5.3 Mechanism & Correction Invariants

| ID  | Invariant                                     | Scope          |
| --- | --------------------------------------------- | -------------- |
| M-1 | Price in [0,1]                                | All prices     |
| M-2 | No-arbitrage projection preserves bounds      | Stage 2        |
| M-3 | Corrections do not worsen loss in expectation | Synthetic only |

**Classification note (binding):**

- **M-3 is E-class only.**
- M-3 is **never** a runtime abort gate.
- M-3 must **not** be enforced via `require_invariant` or any fail-closed mechanism.
- M-3 may be computed and logged **only as an exploratory diagnostic**, reflecting expectation-level behavior over synthetic data.
- Finite-sample violations of M-3 **do not indicate incorrectness**, implementation failure, or contradiction of CAPOPM.pdf.


---

### 5.4 Explicitly Non-Invariants

The following are **not invariants**:

* Posterior concentration
* Regret decay rates
* Dominance over baselines
* Efficiency convergence
* Truth recovery

They may appear **only** as X-class diagnostics.

---

## 6. Failure Classification

| Failure Type           | Meaning                |
| ---------------------- | ---------------------- |
| **Hard Fail**          | Pipeline invalid; stop |
| **Proxy Fail**         | Surrogate inadequate   |
| **Exploratory Signal** | Informational only     |
| **Forbidden**          | Artifact invalidated   |

No failure may be interpreted as contradicting CAPOPM.pdf unless the paper explicitly claims the violated property under the same observables.

---

## 7. Logging & Traceability Requirements

Every run must log:

* Full observable snapshot
* Invariant pass/fail table
* Claim classification ledger
* Random seed
* Config hash

This is enforced by `audit_contracts.py`.

---

## 8. Governance Lock

This contract is **frozen** for Stage B.1–B.2.

Modifications require:

1. Demonstrated audit error
2. Paper-aligned correction
3. Retroactive claim reclassification

---

## 9. Summary Rule (Non-Negotiable)

> **If it is not observable, it does not exist.
> If it exists but is latent, it cannot support a claim.
> If it passes a test, it is not proven.**

---

