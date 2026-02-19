# PRISM_REFACTOR_SPEC.md
# PRISM Refactor Spec (Binding)

**Project:** PRISM — Probabilistic Regime-Integrated Signal Model  
**Purpose of this document:** This is a binding refactor contract for the manuscript rewrite. It defines required edits, constraints, and stop conditions. It is designed for machine execution and human audit.

---

## 0. Non-Negotiables

1. **Canonical rename**
   - Replace all meaningful mentions of `CAPOPM` with `PRISM`.
   - Canonical expansion: **PRISM: Probabilistic Regime-Integrated Signal Model**.
   - **CAPOPM may appear only inside** an appendix section titled **"Project Lineage"**.

2. **PRISM is NOT a pricing model**
   - Remove option-pricing framing as the primary purpose.
   - Remove language implying PRISM "prices options", "produces risk-neutral prices", or is a "pricing engine".
   - PRISM may remain *financially grounded* (market microstructure, financial engineering motivation), but the central output is a **posterior distribution over an event of interest**.

3. **Remove fractional Heston entirely**
   - Delete all mentions of fractional Heston, tempered fractional Heston, Riccati–Volterra derivations, rough volatility kernel discussions, and any dependent exposition.
   - Remove related assumptions, lemmas, theorems, propositions, and citations that exist only to support the fractional Heston story.
   - If any surviving claim previously relied on this material, rewrite it to match the new PRISM workflow (or remove it if it cannot be justified).

4. **Structural prior is Beta(1,1)**
   - Structural anchor prior is: `p ~ Beta(1,1)` unless otherwise explicitly specified in a clearly-labeled optional extension.
   - Any “structural model prior” content must be removed unless reframed as an optional plug-in (clearly separated and non-canonical).

5. **ML is a calibrated signal, not a prior**
   - ML component must be framed as a **signal** that produces an initial probability estimate (or score mapped to probability) that is then **calibrated** and **converted into Beta parameters** in a transparent way.
   - Remove or rewrite the "virtual sample size" story if it is not strictly defensible.
   - Mapping from calibrated probability to Beta parameters must be explained honestly, with explicit assumptions and limitations.
   - Allowed canonical approach (preferred):
     - ML outputs raw probability/score → calibration (logistic/isotonic) → calibrated mean and uncertainty proxy → Beta moment-matching (when feasible).

6. **End-to-end clarity + workflow graphics**
   - Add at least one end-to-end **workflow diagram**, preferably:
     - A **DAG** (TikZ preferred) describing dependencies (data → signals → calibration → fusion → corrections → posterior).
     - If TikZ DAG is too heavy, include a **textual box-and-arrow workflow** (monospace diagram) in the paper.
   - The workflow graphic must match the actual described pipeline (no aspirational steps).

7. **No hallucination**
   - Never introduce citations that are not already in the `.bib` file or otherwise present as real, verifiable sources in the repo.
   - If a citation is broken, unverifiable, or seems hallucinated, **remove it** or mark as **TODO** with a clear note, but do not fabricate replacements.
   - Never claim empirical results that have not been produced and are not traceable to concrete artifacts.

---

## 1. Canonical Workflow (Must Match Manuscript)

The manuscript must represent PRISM as this canonical pipeline:

1. **Data (L3 order flow)**: full depth order book / granular belief data.
2. **Raw belief extraction**: construct initial “anchor beliefs” from raw crowd order flow (explicit definition required).
3. **ML signal generation**: ML provides a probability estimate (or score mapped to probability).
4. **Calibration**: calibrate ML probabilities (or scores) using a transparent calibration method (e.g., logistic calibration, isotonic regression), and quantify calibration uncertainty where feasible.
5. **Fusion**: combine calibrated ML signal with anchor beliefs in a Bayesian-consistent manner, producing intermediate Beta parameters (or an equivalent conjugate representation).
6. **Corrections**: apply bias corrections (behavioral / structural) as explicitly defined transforms that do not overclaim.
7. **Posterior**: output is a posterior distribution over the event probability.

**Important:** If any step is not fully implemented/validated, label it clearly as (i) conceptual, (ii) optional, or (iii) future work. Do not blur these.

---

## 2. Authoritative Theory Reference (Use THIS, Not Legacy Audit Files)

This refactor must treat the following as the canonical “theory map” for what the paper contains and how elements relate:

- **THEORY_APPENDIX.MD** (path: `~/CAPOPM/THEORY_APPENDIX.md`)

THEORY_APPENDIX.MD provides three critical sections that must guide refactoring and preservation:

1) **Proxy Policy Summary Table**  
2) **Comprehensive Element Mapping Table** (Assumptions, Definitions, Remarks, Theorems, Propositions with page references and testing notes)  
3) **Interconnections Table** (dependency graph of elements)

### Explicit deprecation
Do **not** use `claim_table.md` or `borderline_atlas.md` as authoritative for PRISM theory refactor. They are legacy efficiency/smoke-test governance artifacts and do not fully encapsulate the manuscript’s theoretical structure.

### Implementation requirement
During refactor, the agent must:
- cross-check retained/modified/removed elements against THEORY_APPENDIX.MD,
- preserve the mathematical and definitional skeleton unless removal is required by the fractional-Heston deletion or an outright inconsistency,
- log all deviations (see Section 6).

---

## 3. Manuscript Outputs (Deliverables)

The refactor must yield **two compilable documents**:

### A) Main paper (academic-leaning; stats audience)
Must include:
- Framework definition and motivation (financial engineering framing)
- Assumptions and scope (explicit)
- Bayesian workflow (end-to-end)
- Theorems/propositions (only those that remain valid)
- High-level synthetic validation summary (no giant dump of plots)

### B) Supplement / Technical Report
Must include:
- Full simulation study details
- Stress tests
- Diagnostic plots
- Additional proofs (if too bulky for main)

---

## 4. Required Structural Refactor (LaTeX File Split)

The paper must be split into multiple TeX files and compiled from `main.tex`. Target structure is defined in `PRISM_TEX_LAYOUT.md`.

Codex may adjust filenames, but the **split must occur** and the resulting build must work.

---

## 5. Substance Preservation (Anti-Summarization, Binding)

This is a refactor, not a summarization. The agent is forbidden from compressing the work into a minimal “overview” paper.

### 5.1 No-deletion rule (with narrow exceptions)
The agent may not delete substantive theoretical content (definitions, theorems, propositions, lemmas, proofs, key derivations) **except** when one of the following is true:

- **E1: Dependency removal**: the content depends materially on fractional/tempered/rough Heston material that is being deleted.
- **E2: Mathematical invalidity**: the content is blatantly wrong or internally inconsistent and cannot be corrected conservatively.
- **E3: True redundancy**: the content is a duplicate of an equivalent statement retained elsewhere (must point to the replacement location).

If content is removed under E1/E2/E3, it must be recorded in the deletion ledger (Section 6.1) with the reason and replacement location (if any).

### 5.2 Minimum technical mass gates
To prevent “cleanup via collapse,” the agent must satisfy all of the following:

- **Gate A (Theorem-like environment retention):**
  - Count theorem-like environments in the PRISM outputs (main+supp):  
    `theorem`, `lemma`, `proposition`, `definition`, `corollary`  
  - The retained count must be **≥ 70%** of the legacy paper’s count **after excluding** items removed under E1 (fractional-Heston dependency).  
  - The agent must compute and report these counts in `MIGRATION_REPORT.md` (Section 6.2).

- **Gate B (Displayed equation retention):**
  - Count displayed math blocks (`equation`, `align`, `\[...\]`) in PRISM outputs (main+supp).
  - Retained count must be **≥ 70%** of legacy count after excluding E1 removals.
  - Report counts in `MIGRATION_REPORT.md`.

- **Gate C (Length floor):**
  - The combined PRISM main+supp PDF page count must be **≥ 80%** of the legacy paper’s PDF page count after excluding the fractional-Heston section(s).  
  - If fractional-Heston removal substantially changes page count, the agent must justify the delta with a structured table in `MIGRATION_REPORT.md`.

If any gate fails, the agent must continue working until the gates pass.

### 5.3 Preserve core theoretical skeleton (from THEORY_APPENDIX.MD)
Unless removed under E1/E2/E3, the agent must preserve:
- the majority of Assumptions, Definitions, and Theorems enumerated in THEORY_APPENDIX.MD,
- their dependency order (or explicitly justify a reordering),
- and the conceptual integrity of the framework.

---

## 6. Audit Artifacts (Required Files)

To make the refactor verifiable and prevent silent deletions, the agent must generate the following files under `paper/prism/`:

### 6.1 `DELETION_LEDGER.md` (required)
A table listing every removed/merged/superseded element with:
- legacy identifier (e.g., “Thm 12”, “Def 4”, label if present)
- legacy page (if known)
- action: removed / merged / moved / rewritten
- reason: E1 / E2 / E3
- destination replacement (file + label) or “none”
- brief rationale (2–4 lines)

### 6.2 `MIGRATION_REPORT.md` (required)
Must include:
- mapping from legacy sections to new files (high-level)
- environment counts (Gate A/B), legacy vs PRISM (with E1 exclusions documented)
- page count comparison (Gate C)
- list of any theorem/proposition renumbering or label remapping
- note of any major conceptual reorderings (and why)

### 6.3 `RESULTS_CITATION_INDEX.md` (required)
If the manuscript or supplement makes **any** empirical statement (including “observed,” “passed,” “failed,” metric values, comparisons), it must:
- cite the exact artifact path(s) supporting it (e.g., `results/.../metrics_aggregated.csv`)
- describe the extraction method briefly
- include the manuscript location where the claim is made (file + section)

**If an empirical statement cannot be tied to artifacts, it must be removed or rephrased as hypothetical/future work.**

### 6.4 `CHANGELOG_PRISM_REFACTOR.md` (required)
Already required; must additionally:
- summarize Gate A/B/C outcomes (final values)
- summarize major removals and why
- summarize how THEORY_APPENDIX.MD was used

---

## 7. Quality Gates (Hard Stop Conditions)

Codex must not stop until all of the following are true:

1. `latexmk -pdf` (or equivalent) builds **without errors** for both main paper and supplement.
2. No undefined references (no “??”), no broken citations, no missing figures.
3. Terminology is consistent: PRISM, anchor beliefs, calibrated ML signal, corrections, posterior.
4. No remaining fractional Heston references, text or equations, in active PRISM builds.
5. CAPOPM appears only in `appendix/a01_project_lineage.tex`.
6. Every theorem/proposition/lemma/definition is either:
   - kept and made consistent, or
   - removed under E1/E2/E3 and logged in `DELETION_LEDGER.md`, or
   - moved to supplement with clear destination location.
7. Workflow graphic exists and matches pipeline.
8. Writing is tightened: grammar, spelling, and redundant passages removed.
9. Substance Preservation Gates (Section 5.2) all pass and are documented in `MIGRATION_REPORT.md`.
10. Any empirical claim is traceable to artifacts via `RESULTS_CITATION_INDEX.md` or removed/rephrased.

---

## 8. Mathematical Editing Rules

- Do not change fundamental arguments unless required to remove inconsistencies introduced by the prior refactor.
- If an argument is blatantly wrong, correct it conservatively:
  - prefer tightening assumptions, clarifying conditions, or weakening claims,
  - do not invent new theorems,
  - do not introduce untested empirical claims.
- If a proof relies on deleted structural-prior machinery, rewrite it to rely only on the PRISM workflow (conjugacy / calibration / transformations), or remove it under E1 and log it.

---

## 9. Citation Rules

- Use only citations present in the `.bib` file or otherwise already provided in the project.
- If a citation is suspected hallucinated or cannot be validated:
  - remove it, and if needed leave a TODO note in comments.
- Do not add new references from memory.

---

## 10. Tone Requirements

- Academic, clear, and restrained.
- Honest about what is tested vs conceptual.
- Financial engineering framing without marketing language.
- No overclaiming, no “state-of-the-art” claims.

---
