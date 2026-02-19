# DELETION_LEDGER

This ledger records theorem-like removals performed under the PRISM refactor contract.

Legend:
- `E1`: dependency removal (fractional/option-pricing dependent)
- `E2`: mathematical invalidity
- `E3`: true redundancy

| Legacy Identifier | Legacy Location | Action | Reason | Destination Replacement | Rationale |
|---|---|---|---|---|---|
| Lemma: Existence and uniqueness of the variance process | `_legacy/main.tex:427` | removed | E1 | none | Depends on fractional-Volterra variance dynamics removed from canonical PRISM workflow. |
| Lemma: Positivity of variance | `_legacy/main.tex:441` | removed | E1 | none | Depends on deleted fractional structural-prior SDE block. |
| Proposition: Existence of a continuous density for `S_T` | `_legacy/main.tex:453` | removed | E1 | none | Structural-density existence result tied to deleted fractional model block. |
| Theorem: Fractional Parameter Sensitivity of Structural Tail Probabilities | `_legacy/main.tex:581` | removed | E1 | none | Explicit fractional/Heston sensitivity theorem removed by contract. |
| Lemma: Integrability of the structural density | `_legacy/main.tex:742` | removed | E1 | none | Integrability argument depends on deleted structural density machinery. |
| Proposition: Structural prior as parimutuel prior odds | `_legacy/main.tex:770` | rewritten | E1 | `sections/04_data_and_anchor_beliefs.tex` | Replaced by canonical flat structural anchor `Beta(1,1)` and anchor-evidence update. |
| Theorem: PRISM-Implied Risk-Neutral CDF, Density, and Pricing Kernel | `_legacy/main.tex:2595` | removed | E1 | none | Option-pricing/risk-neutral theorem removed; PRISM is not a pricing model. |
| Theorem: PRISM Kernels and Kernel Shapes | `_legacy/main.tex:2770` | removed | E1 | none | Pricing-kernel stylized-facts theorem removed from active scope. |
| Lemma: Positivity and Normalization of the Kernel | `_legacy/main.tex:2878` | removed | E1 | none | Kernel positivity lemma tied to removed pricing-kernel story. |
| Definition: Drift Adjustment and No-Arbitrage | `_legacy/main.tex:2914` | removed | E1 | none | Derivative-pricing no-arbitrage definition removed from canonical PRISM scope. |
| Theorem: Kernel Regularization and No-Arbitrage Preservation | `_legacy/main.tex:2964` | removed | E1 | none | Pricing-kernel regularization theorem removed with pricing engine framing. |
| Lemma: Posterior Mean and Variance (Phase 5 block) | `_legacy/main.tex:3030` | moved | E1 | `sections/07_posterior_and_outputs.tex` | Retained posterior moments in inference context; removed pricing-dependent framing. |
| Proposition: Posterior Predictive Distribution of Digital Payoff | `_legacy/main.tex:3061` | removed | E1 | none | Digital-payoff pricing proposition removed from active manuscript scope. |
| Theorem: Arbitrage-Free YES/NO Pricing | `_legacy/main.tex:3095` | removed | E1 | none | Contract-pricing theorem removed from canonical PRISM inference pipeline. |
| Proposition: Monotonicity in YES Votes | `_legacy/main.tex:3117` | rewritten | E1 | `sections/07_posterior_and_outputs.tex` | Monotonic posterior-update interpretation retained; pricing statement removed. |
| Proposition: Continuity | `_legacy/main.tex:3131` | rewritten | E1 | `sections/08_theory.tex` | Stability continuity retained as posterior-map regularity claim. |
| Definition: Stacked Mixture Posterior over `p` (pricing phase variant) | `_legacy/main.tex:3191` | moved | E1 | `supplement/s06_extended_legacy_derivations.tex` | Retained mixture posterior language in inference setting only. |
| Proposition: Posterior Mean and Predictive under the Mixture | `_legacy/main.tex:3220` | moved | E1 | `supplement/s06_extended_legacy_derivations.tex` | Retained as posterior-mixture summary without derivative-pricing interpretation. |
| Definition: Moment-Matched Single-Beta Approximation | `_legacy/main.tex:3278` | moved | E1 | `sections/05_ml_signal_and_calibration.tex` | Preserved as calibrated ML uncertainty transport via Beta moment matching. |
| Theorem: No Unimodal Beta Can Uniformly Approximate a Strongly Multimodal Mixture | `_legacy/main.tex:3315` | moved | E1 | `supplement/s06_extended_legacy_derivations.tex` | Retained as inference limitation theorem; removed phase-5 pricing framing. |
| Lemma: Risk Monotonicity | `_legacy/main.tex:3430` | removed | E1 | none | Legacy risk/price monotonicity target removed with pricing engine framing. |
| Proposition: No-Arbitrage Identity Preserved | `_legacy/main.tex:4193` | removed | E1 | none | Explicit no-arbitrage pricing identity removed from active canonical PRISM scope. |
| Theorem: Arbitrage-Freeness of CAPOPM Pricing | `_legacy/main.tex:5871` | removed | E1 | none | Deleted as pricing-only theorem inconsistent with PRISM claim boundaries. |

