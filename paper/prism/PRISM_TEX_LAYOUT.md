# PRISM_TEX_LAYOUT.md
# PRISM LaTeX Layout & Refactor Map (Binding)

**Purpose:** Define the canonical file/folder layout for the PRISM manuscript and provide a migration plan from the current monolithic TeX source into a modular, auditable structure. This document is binding for Codex during refactor.

---

## 0. Canonical Directory Layout

All PRISM paper assets must live under:

paper/prism/
├── main.tex
├── supplement.tex
├── sections/
│ ├── 00_abstract.tex
│ ├── 01_introduction.tex
│ ├── 02_scope_and_positioning.tex
│ ├── 03_notation_and_problem_setup.tex
│ ├── 04_data_and_anchor_beliefs.tex
│ ├── 05_ml_signal_and_calibration.tex
│ ├── 06_fusion_and_corrections.tex
│ ├── 07_posterior_and_outputs.tex
│ ├── 08_theory.tex
│ ├── 09_validation_summary.tex
│ ├── 10_discussion_limitations_future_work.tex
│ └── 11_conclusion.tex
├── supplement/
│ ├── s01_simulation_design.tex
│ ├── s02_stress_tests.tex
│ ├── s03_diagnostics_and_plots.tex
│ ├── s04_additional_proofs.tex
│ └── s05_additional_tables.tex
├── appendix/
│ ├── a01_project_lineage.tex
│ ├── a02_proofs.tex
│ └── a03_extra_results.tex
├── figures/
│ ├── workflow/
│ │ ├── prism_workflow_dag.tikz
│ │ ├── prism_workflow_boxarrows.tex
│ │ └── prism_workflow_dag.pdf (if compiled/exported)
│ └── (other figures...)
├── bibliography/
│ └── references.bib
├── CHANGELOG_PRISM_REFACTOR.md
├── PRISM_REFACTOR_SPEC.md
├── PRISM_STYLE_GUIDE.md
├── PRISM_CLAIM_BOUNDARIES.md
└── PRISM_TEX_LAYOUT.md



**Notes**
- `main.tex` compiles the main paper.
- `supplement.tex` compiles the technical supplement.
- CAPOPM is allowed **only** in `appendix/a01_project_lineage.tex`.
- Fractional Heston must not appear anywhere in PRISM files.

---

## 1. Compilation Interfaces

### 1.1 `main.tex` must:
- Define document class, packages, macros, theorem environments.
- `\input{sections/...}` in the order listed below.
- `\appendix` then include appendix files (project lineage, proofs, extra results) **only if** you want appendices inside the main PDF.  
  - If you prefer appendices in the supplement only, keep the main paper lean and move bulky proofs to `supplement.tex`.

### 1.2 `supplement.tex` must:
- Be independently compilable.
- Reuse shared macros via a `tex/` shared file *if needed* (optional).
- Include simulation design, stress tests, diagnostics, and long proofs.

**Hard requirement:** both PDFs compile without errors and with no missing references/citations.

---

## 2. Section Objectives (What Each File Must Contain)

This is the canonical narrative order, replacing “Phase 1–8” framing.

### `00_abstract.tex`
- Tight abstract describing PRISM as a Bayesian workflow for posterior inference over an event of interest using L3 order flow + calibrated ML signal + corrections.

### `01_introduction.tex`
- Motivation grounded in financial engineering.
- The “why”: belief aggregation under noisy/incomplete information; order flow as belief data.

### `02_scope_and_positioning.tex`
- Explicitly state: PRISM is not a pricing model.
- Explain what problems it *does* address (posterior belief extraction and uncertainty).
- Explain deliverables and limitations.

### `03_notation_and_problem_setup.tex`
- Define event of interest, probability `p`, datasets, time indexing.
- Define key objects: order flow, anchor beliefs, ML signal, calibration, fusion, corrections, posterior.
- Keep assumptions minimal and explicit.

### `04_data_and_anchor_beliefs.tex`
- Define L3 order book inputs (at a conceptual level if needed).
- Define “anchor beliefs” precisely:
  - what it is (statistic/distribution)
  - what it represents
  - how computed from raw order flow
- Include any modeling assumptions needed for anchor belief extraction.

### `05_ml_signal_and_calibration.tex`
- ML is a signal generator (not a prior).
- Define ML output: probability estimate or score.
- Define calibration:
  - method (logistic calibration / isotonic)
  - calibration metrics (Brier, log score, reliability diagram references if already in project)
- Define how calibrated outputs are represented probabilistically.

### `06_fusion_and_corrections.tex`
- Fusion step: combine anchor beliefs + calibrated ML info.
- Must result in a coherent representation; prefer Beta parameterization where consistent with manuscript.
- Corrections:
  - behavioral distortions (if used)
  - structural distortions (liquidity/whales) (if used)
- Corrections must be stated as transforms with constraints and limitations.

### `07_posterior_and_outputs.tex`
- Posterior definition and closed-form update if Beta-Binomial is still used.
- Output objects:
  - posterior mean, credible intervals
  - diagnostics
  - any posterior predictive objects (only if consistent with “not pricing model”).

### `08_theory.tex`
- Keep only theorems that are still valid under:
  - Beta(1,1) structural prior
  - ML as calibrated signal
  - anchor belief fusion + correction transforms
- Remove any theorem relying on fractional Heston or option pricing kernels.
- Tight proofs or move long proofs to appendix/supplement.

### `09_validation_summary.tex`
- High-level summary of synthetic validation:
  - scenarios
  - metrics
  - key findings
  - limitations
- Keep it short and interpretable; refer to supplement for details.

### `10_discussion_limitations_future_work.tex`
- Honest discussion of limitations.
- Future work includes:
  - improved anchor extraction
  - richer correction models
  - empirical studies (if not done)
  - optional plug-ins (e.g., structural priors), clearly labeled non-canonical.

### `11_conclusion.tex`
- Summary and contributions (no overclaiming).

---

## 3. Workflow Diagram Requirements

At minimum, include a workflow graphic in the main paper:

### Preferred: TikZ DAG
- File: `figures/workflow/prism_workflow_dag.tikz`
- Include via `\input{figures/workflow/prism_workflow_dag.tikz}` inside a `figure` environment.

### Backup: Textual box-and-arrow diagram
- File: `figures/workflow/prism_workflow_boxarrows.tex`
- Use monospace text with `\ttfamily` or `verbatim`.

**Diagram content must match canonical pipeline:**
L3 order flow → anchor beliefs → ML signal → calibration → fusion → corrections → posterior.

Optional nodes may be shown only if labeled "optional" or "future".

---

## 4. Appendix Placement Rules

### Main paper appendices (optional)
- `appendix/a01_project_lineage.tex` (CAPOPM history only)
- `appendix/a02_proofs.tex` (if short)
- `appendix/a03_extra_results.tex` (if short)

### Supplement appendices (recommended for bulk)
- Put heavy proofs in `supplement/s04_additional_proofs.tex`.
- Put full results in `supplement/s03_diagnostics_and_plots.tex` and `supplement/s05_additional_tables.tex`.

**Hard constraint:** CAPOPM appears only in `appendix/a01_project_lineage.tex`.

---

## 5. Bibliography Rules

- Only one authoritative bibliography file:
  - `bibliography/references.bib`
- All `\cite{}` keys must exist in this file.
- If a citation is broken/unverifiable:
  - remove the cite or comment it with TODO
  - do not invent replacement citations

---

## 6. Macro & Environment Rules

Codex must consolidate macros and theorem environments into one place:

Recommended:
- In `main.tex` and `supplement.tex`, define:
  - theorem environments
  - notation macros
  - reference macros
- If shared macros get large, create:
  - `tex/prism_preamble.tex` (optional)
  - and input it from both.

**Avoid duplicated macro definitions** across files.

---

## 7. Migration Instructions (From Monolith to Modular)

Codex should follow this migration sequence:

1. **Create target folder structure** under `paper/prism/`.
2. **Copy the current Overleaf sources** into `paper/prism/_legacy/` unchanged (for audit trail).
3. Create new `main.tex` and `supplement.tex` shells.
4. Split the monolithic content into section files following Section 2 mapping.
5. Remove all fractional Heston content during the split (do not carry it into new files).
6. Rewrite narrative flow to match PRISM positioning and Scope document.
7. Insert workflow diagram.
8. Fix:
   - grammar/spelling
   - LaTeX syntax
   - references/citations
9. Build, iterate until clean.
10. Write `CHANGELOG_PRISM_REFACTOR.md` summarizing all major changes.

---

## 8. Definition of “Done” (Stop Conditions)

This refactor is complete only when:

- `main.tex` compiles without errors/warnings that indicate broken refs/cites.
- `supplement.tex` compiles without errors/warnings that indicate broken refs/cites.
- No fractional Heston content exists in `paper/prism/` (including legacy removed from compiled build paths).
- CAPOPM appears only in `appendix/a01_project_lineage.tex`.
- Workflow diagram exists and matches pipeline.
- The paper reads as a coherent PRISM framework paper (not CAPOPM, not pricing model).
- Writing is tightened and consistent with PRISM style/claim boundary docs.
