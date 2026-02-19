# PRISM Refactor Changelog

## Snapshot and Audit Trail
- Created `paper/prism/_legacy/` and copied the previous manuscript sources unchanged:
  - `paper/prism/main.tex`
  - `paper/prism/phase1_refactor.tex`
  - `paper/prism/phase2_hybrid_fusion.tex`
  - `paper/prism/phase3_parimutuel_mechanism.tex`
  - `paper/prism/phase6_behavioral_corrections.tex`
  - `paper/prism/phase7_simulations.tex`
  - `paper/prism/phase8_theoretical_results.tex`
  - `paper/prism/literature_review_conclusion.tex`
  - `paper/prism/Refs.bib`
  - `paper/prism/rnn-diagram.tex`

## Structural Migration
- Replaced monolithic active manuscript with modular structure:
  - `paper/prism/main.tex`
  - `paper/prism/supplement.tex`
  - `paper/prism/tex/prism_preamble.tex`
  - `paper/prism/sections/*.tex`
  - `paper/prism/supplement/*.tex`
  - `paper/prism/appendix/*.tex`
  - `paper/prism/figures/workflow/*`
  - `paper/prism/bibliography/references.bib`

## Major Content Rewrites
- Reframed manuscript as a Bayesian workflow with posterior inference output.
- Replaced phase-based narrative with section-based flow.
- Set canonical structural prior to `Beta(1,1)`.
- Rewrote ML component as calibrated signal with conservative uncertainty transport and Beta moment matching.
- Reworked fusion and correction exposition into explicit transformation maps.
- Reduced main-paper empirical section to high-level audit-aligned summary and moved extended material to supplement.

## Major Deletions and Claim Tightening
- Removed legacy structural-prior machinery and dependent exposition from the active manuscript.
- Removed active references to deprecated model blocks and optional downstream valuation framing.
- Tightened theory statements to claims directly supported by the revised workflow assumptions.

## Theorem and Proof Updates
- Kept only workflow-consistent propositions/theorems in `sections/08_theory.tex`.
- Added proof details in `appendix/a02_proofs.tex` and `supplement/s04_additional_proofs.tex`.
- Removed broad claims that required assumptions not maintained in the revised scope.

## Citation and Bibliography Changes
- Consolidated active bibliography to `paper/prism/bibliography/references.bib`.
- Removed unused and out-of-scope citation entries from active bibliography.
- Kept citations limited to keys already present in repository sources.

## Workflow Figures Added
- `paper/prism/figures/workflow/prism_workflow_dag.tikz`
- `paper/prism/figures/workflow/prism_workflow_boxarrows.tex`

## How to Compile
From `paper/prism/`:
```powershell
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```
If `latexmk` is unavailable:
```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

pdflatex supplement.tex
bibtex supplement
pdflatex supplement.tex
pdflatex supplement.tex
```

## Final Gate Outcomes (A/B/C)
- Gate A (theorem-like retention):
  - Legacy total: 84
  - E1 exclusions: 23
  - Legacy adjusted: 61
  - PRISM retained: 68
  - Threshold: 43
  - Result: PASS
- Gate B (displayed equation retention):
  - Legacy total: 435
  - E1 exclusions: 98
  - Legacy adjusted: 337
  - PRISM retained: 337
  - Threshold: 236
  - Result: PASS
- Gate C (page floor):
  - Legacy PDF pages: 118
  - E1 page exclusions: 23
  - Legacy adjusted: 95
  - PRISM pages (`main.pdf` + `supplement.pdf`): 77
  - Threshold: 76
  - Result: PASS

## Theory-Appendix Usage
- Used `THEORY_APPENDIX.MD` as canonical dependency map for assumptions/definitions/theorem families.
- Retained and expanded non-E1 posterior/concentration/robustness/dependence families in active PRISM sections and supplement.
- Removed fractional/pricing-dependent theorem chain under E1 and documented each removal in `DELETION_LEDGER.md`.
