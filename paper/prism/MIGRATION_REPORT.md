# MIGRATION_REPORT

## 1. Legacy-to-PRISM Section Mapping

| Legacy Source Block | PRISM Destination | Migration Note |
|---|---|---|
| Monolithic intro + phase framing (`_legacy/main.tex` pre-Phase 1) | `sections/01_introduction.tex`, `sections/02_scope_and_positioning.tex`, `sections/03_notation_and_problem_setup.tex` | Reframed to canonical PRISM scope and workflow language. |
| Phase 1 (structural fractional block) | removed (E1) | Deleted per hard constraint: no fractional/Heston/Volterra active content. |
| Phase 2 + Phase 3 prior/likelihood setup | `sections/04_data_and_anchor_beliefs.tex`, `sections/05_ml_signal_and_calibration.tex`, `sections/06_fusion_and_corrections.tex`, `sections/07_posterior_and_outputs.tex`, `supplement/s06_extended_legacy_derivations.tex` | Rewritten around flat structural anchor `Beta(1,1)` and calibrated ML signal transport. |
| Phase 4 conjugate update derivations | `sections/07_posterior_and_outputs.tex`, `supplement/s06_extended_legacy_derivations.tex` | Core conjugate machinery retained and expanded in supplement. |
| Phase 5 posterior-predictive pricing block | largely removed (E1), selective posterior-mixture fragments moved to `supplement/s06_extended_legacy_derivations.tex` | Pricing-engine framing removed; inference content retained where defensible. |
| Phase 6 corrections/robustness block | `sections/06_fusion_and_corrections.tex`, `sections/08_theory.tex`, `supplement/s06_extended_legacy_derivations.tex` | Behavioral/structural transforms retained and expanded. |
| Phase 7 simulation/stress material | `sections/09_validation_summary.tex`, `supplement/s01_simulation_design.tex`, `supplement/s02_stress_tests.tex`, `supplement/s03_diagnostics_and_plots.tex`, `supplement/s05_additional_tables.tex`, `supplement/s06_extended_legacy_derivations.tex` | Main text kept high-level; supplement carries detail. |
| Phase 8 theorem/proof block | `sections/08_theory.tex`, `appendix/a02_proofs.tex`, `supplement/s04_additional_proofs.tex`, `supplement/s06_extended_legacy_derivations.tex` | Majority retained; pricing-only theorems removed under E1. |
| Historical naming and project continuity | `appendix/a01_project_lineage.tex` | Only active location where CAPOPM appears. |

## 2. Substance Preservation Gates

Counting method:
- Legacy counts: regex scan of `paper/prism/_legacy/main.tex`.
- PRISM counts: regex scan over active TeX files under `paper/prism/` excluding `_legacy`.
- E1 exclusions (count gate basis): Phase 1, Phase 5, subsection 6.6, subsection 8.10 from legacy manuscript.

### Gate A: Theorem-like Environments

| Metric | Count |
|---|---:|
| Legacy theorem-like total | 84 |
| E1-excluded theorem-like | 23 |
| Legacy adjusted denominator | 61 |
| 70% threshold (`ceil(0.70*61)`) | 43 |
| PRISM retained theorem-like | 68 |
| Gate A result | **PASS** |

### Gate B: Displayed Equations

| Metric | Count |
|---|---:|
| Legacy displayed-equation total | 435 |
| E1-excluded displayed equations | 98 |
| Legacy adjusted denominator | 337 |
| 70% threshold (`ceil(0.70*337)`) | 236 |
| PRISM retained displayed equations | 337 |
| Gate B result | **PASS** |

### Gate C: Page Count Floor

Page-count method:
- Legacy page count from `docs/CAPOPM_paper.pdf` (`pdfinfo`): 118 pages.
- E1 page exclusions identified from section starts in legacy PDF:
  - Phase 1: pages 5--12 (8 pages)
  - Phase 5: pages 40--52 (13 pages)
  - subsection 6.6: page 61 (1 page)
  - subsection 8.10: page 85 (1 page)
- Total E1 page exclusion: 23 pages.

| Metric | Count |
|---|---:|
| Legacy pages (raw) | 118 |
| E1-excluded pages | 23 |
| Legacy adjusted denominator | 95 |
| 80% threshold (`ceil(0.80*95)`) | 76 |
| PRISM pages (`main.pdf` + `supplement.pdf`) | 77 |
| Gate C result | **PASS** |

## 3. Renumbering and Label Remapping Notes

- The active manuscript now numbers theorem environments from modular files (`sections/08_theory.tex` and supplement modules) rather than monolithic phase-local numbering.
- Legacy `prop` usage is supported in active preamble via explicit environment alias (`tex/prism_preamble.tex`) to preserve imported theorem blocks.
- Cross-file label usage was normalized where legacy references depended on removed sections.

## 4. Major Conceptual Reorderings

- Canonical pipeline ordering is now explicit and fixed in main text: L3 order flow -> anchor beliefs -> ML signal -> calibration -> fusion -> corrections -> posterior.
- Structural prior is canonicalized to `Beta(1,1)`; structural-fractional machinery was deleted as non-canonical.
- ML treatment is recast as calibrated signal transport with conservative Beta moment-matching, not as prior replacement.
- Pricing-engine claims were removed from active claim scope; posterior inference remains the primary output contract.

## 5. Theory-Appendix Alignment Notes

`THEORY_APPENDIX.MD` was used as the canonical dependency map for retained vs removed theory elements.
- Retained families: posterior consistency/stability, concentration, mixture robustness, dependence-aware limits, correction-map regularity.
- Removed families (E1): fractional structural-prior and pricing-kernel/arbitrage-centric theorem chain.
- All E1 removals are logged in `DELETION_LEDGER.md`.
