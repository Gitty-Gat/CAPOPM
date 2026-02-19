# PRISM_STYLE_GUIDE.md
# PRISM Style Guide (Binding)

**Goal:** Produce a readable, academic-leaning manuscript for a mixed audience (stats first), with a financial engineering theme. The writing must be honest, non-pompous, and internally consistent.

---

## 1. Voice and Tone

**Write like:** an academic methods paper with an engineering mindset.  
**Avoid:** pitch-deck language.

### Do
- Use simple sentence structure where possible.
- Prefer concrete nouns and verbs.
- State assumptions explicitly.
- Separate "what PRISM does now" vs "possible extensions".

### Do not
- Use grandiose claims ("revolutionary", "breakthrough", "guarantees dominance").
- Overuse buzzwords ("paradigm", "next-generation", "unified intelligence").
- Use vague claims ("robust", "optimal") without definitions.

---

## 2. Terminology Glossary (Canonical)

Use these terms consistently:

- **PRISM**: the full workflow
- **Event of interest**: the binary (or clearly specified) outcome being inferred
- **Anchor beliefs**: beliefs extracted from raw crowd order flow (must be defined precisely)
- **ML signal**: an ML-generated probability estimate or score prior to calibration
- **Calibrated ML signal**: ML output after calibration and uncertainty characterization
- **Fusion**: the step that combines anchor beliefs and calibrated ML signal
- **Corrections**: bias correction transforms (behavioral/structural), explicitly defined
- **Posterior**: final posterior distribution over event probability

Explicitly avoid:
- “pricing model”
- “option pricing engine”
- “risk-neutral pricing mechanism”
unless clearly labeled as **out of scope** or **future extension**.

---

## 3. Section Design Principles

Each major section must begin with:
1. A 2–4 sentence **purpose statement**
2. A 3–6 bullet **what you will learn** list (short)

Each section must end with:
- a 2–4 bullet **what this section established** list

---

## 4. Mathematical Presentation

- Use consistent notation:
  - If the event probability is `p`, keep it `p` everywhere.
  - If intermediate estimates exist, use clear subscripts:
    - `p_anchor`, `p_ml_raw`, `p_ml_cal`, `p_fused`, `p_post`
- Whenever you introduce a transformation, include:
  - the input
  - the output
  - constraints (domain/range)
  - interpretation

Prefer “small theorems” over long, sweeping claims. If something is not proven, label it.

---

## 5. Claims and Qualifiers

Use the following claim labels:
- **Definition**: formally defines a quantity
- **Proposition**: a precise claim with proof or clear justification
- **Lemma**: supporting claim
- **Theorem**: only if genuinely substantive and supported
- **Remark**: interpretation, limitations, or intuition
- **Assumption**: explicit modeling assumption
- **Conjecture / Hypothesis**: permitted only in “Future Work” or clearly marked subsections

Never state conjectures as propositions.

---

## 6. Empirical / Synthetic Results Writing

Main paper:
- provide **high-level** summary:
  - what scenarios were tested
  - what metrics were used
  - what the results suggest
  - what remains unresolved
- keep plots minimal and interpretable

Supplement:
- dump detailed plots and stress tests
- include reproducibility notes

Avoid “we show superiority” unless the test design supports it.

---

## 7. Workflow Graphic Requirements

Include at least one of:
1. **DAG** (preferred): nodes = pipeline stages, edges = dependency flow
2. **Textual workflow**: monospace boxes with arrows

The graphic must match the described pipeline. No aspirational components.

---

## 8. Editing Hygiene

- Remove repetition aggressively.
- Fix grammar/spelling.
- Remove stale paragraphs that refer to deleted priors or pricing framing.
- Ensure all references compile (`\ref`, `\eqref`, citations).

---

## 9. Appendix “Project Lineage”

This appendix may mention CAPOPM, fractional Heston, and prior history, but must:
- clearly state these are historical
- state what changed and why
- avoid reintroducing removed technical dependencies into the main paper
