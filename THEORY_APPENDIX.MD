# **THEORY_APPENDIX.md**

**CAPOPM Theoretical Elements Appendix**  
**Status:** Canonical Reference (extracted from requirements.txt; cross-references CAPOPM.pdf)  
**Purpose:** Comprehensive tabular mappings for theorems, propositions, assumptions, definitions, remarks, and related policies. Provides full details on claims, metrics, success definitions, pipeline implementations, allowed/forbidden proxies, min sample regimes, and interconnections. For use by AI Agent (Codex) in Stage B to ensure paper alignment without overinterpretation.

---

## Section 1: Proxy Policy Summary Table

This table summarizes allowed/forbidden metrics and min sample regimes per theorem/proposition group.

| Theorem/Prop Group | Allowed Metrics | Forbidden Metrics | Min Sample Regime |
|--------------------|-----------------|-------------------|-------------------|
| Sensitivity/Robustness | Divergence bounds, slopes | Dominance over baselines | Finite n>100 + CI |
| Equilibrium | Mixing rates | Uniform indep. | Asymptotic n-grid |
| Posterior | Concentration trends | Uniform consistency | Asymptotic + bootstraps |
| Kernel/Arbitrage | Violation rates | Puzzle resolution | Finite smoke |
| Mixture/Impossibility | Mode divergences | Uniform approx. | Untestable small-n |
| Propositions | Decompositions | Regret dominance | Mixed finite/asymp |

---

## Section 2: Comprehensive Element Mapping Table

This table covers all assumptions (1-11), definitions (1-10), remarks (1-32), theorems (1-36), and propositions (1-19). Each entry includes: Item Type, Number, Page, Claim in Plain Text, Metrics to Test, Definition of Success, Pipeline Implementation.

| Item Type | Number | Page | Claim in Plain Text | Metrics to Test | Definition of Success | Pipeline Implementation |
|-----------|--------|------|---------------------|-----------------|-----------------------|-------------------------|
| Assumption | 1 | 7 | Structural prior and market liquidity assumptions for initial Beta parameters. | Simulation of prior mean convergence; Brier score on synthetic markets. | Prior mean equals true probability in idealized case. | Phase 3: Used in hybrid prior construction to set α0, β0. |
| Assumption | 2 | 45 | Novikov condition for martingale representation. | Check martingale property in variance dynamics; variance bounds. | Variance process is a martingale under measure change. | Phase 1: Integrated into fractional Heston model for volatility dynamics. |
| Assumption | 3 | 55 | Finite structural distortion regimes with prior weights, pseudo-count corrections, boundedness, and admissibility. | Mixture log-likelihood comparison; regime weight stability. | Posterior is a finite mixture of Betas with positive parameters. | Phase 6: Applied in Stage 2 for nonlinear structural adjustments. |
| Assumption | 4 | 67 | Regime-switching dynamics for bias and distortion parameters as hidden Markov chain. | HMM fit quality (AIC/BIC); filtering accuracy on simulated data. | Switching Beta prior pulls distortion toward regime baseline. | Phase 6: Used in dynamic bias state for time-varying corrections. |
| Assumption | 5 | 88 | Moderate dependence via effective sample size n* for concentration bounds. | Hoeffding inequality validation; effective n* vs. raw n ratio. | Posterior mean concentrates at sub-Gaussian rate. | Phase 8: Incorporated in finite-sample concentration for robustness checks. |
| Assumption | 6 | 90 | Mixture stability under perturbations with component Lipschitz and weight stability. | Total variation distance between mixtures; perturbation sensitivity. | Mixture inherits robustness from components and weights. | Phase 5: Applied to multimodal posteriors for divergence bounds. |
| Assumption | 7 | 96 | Base posterior robustness in Wasserstein distance. | W1 distance on perturbed datasets; stability plots. | W1 bound holds for base posterior mapping. | Phase 8: Used in metric-based robustness for nonlinear updates. |
| Assumption | 8 | 96 | Lipschitz nonlinear adjustment map. | Lipschitz constant estimation; gradient norms. | Adjustment is globally Lipschitz. | Phase 6: Ensures stability in nonlinear distortion corrections. |
| Assumption | 9 | 97 | Smooth monotone nonlinear adjustment with bounded derivative. | Hellinger distance pre/post transformation; derivative bounds. | Hellinger stability under smooth maps. | Phase 8: For Hellinger robustness in smooth updates. |
| Assumption | 10 | 106 | Ergodic regime switching with regularity, identifiability, and finite moments. | Asymptotic variance Vφ computation; BvM approximation error. | Posterior is asymptotically normal under ergodic HMM. | Phase 6: In dynamic regime-switching for bias layers. |
| Assumption | 11 | 108 | Fast regime switching with vanishing dwell time. | Dwell time statistics; estimation error lower bounds. | No uniform consistency possible. | Phase 8: Defines limits for impossibility results in fast switching. |
| Definition | 1 | 5 | Hybrid prior as mixture of structural and ML priors. | Prior mean and variance matching. | Combines liquidity-based structural prior with ML prior. | Phase 3: Defines initial Beta prior for pipeline start. |
| Definition | 2 | 49 | Multimodal posterior as finite mixture of Betas. | Mixture fit vs. single Beta (KL divergence). | Captures multimodal beliefs from stratified data. | Phase 5: Used for stratified estimation in multimodal cases. |
| Definition | 3 | 49 | Moment-matched single Beta for mixture approximation. | Mean/variance match error; approximation quality. | Matches mean and variance of mixture. | Phase 5: Approximates mixtures for simpler computation. |
| Definition | 4 | 55 | Stage 2 nonlinear structural adjustment as regime mixture. | Regime weight posterior probabilities. | Marginalizes over latent regimes for Beta mixture. | Phase 6: Implements nonlinear corrections in Stage 2. |
| Definition | 5 | 67 | Dynamic bias state and emissions in HMM. | Filtering distribution accuracy. | Joint hidden state for bias evolution. | Phase 6: For online updating in dynamic bias layer. |
| Definition | 6 | 68 | Dynamic bias estimates for Stage 2 using filtered expectation. | Filtered expectation convergence. | Uses HMM filtering for time-indexed corrections. | Phase 6: Computes time-varying pseudo-counts. |
| Definition | 7 | 75 | Arbitrage-free simplex for YES/NO prices. | Sum and positivity checks. | One-dimensional probability simplex enforcing no-arbitrage. | Phase 7: Projects prices to ensure arbitrage-freeness. |
| Definition | 8 | 76 | Positivity rectification and normalization for projection. | Projection stability; Lipschitz bounds. | Enforces positivity and renormalizes to simplex. | Phase 7: Final step in price calibration. |
| Definition | 9 | 96 | Nonlinear posterior update via pushforward. | Pushforward measure properties. | Adjusted posterior as image under nonlinear map. | Phase 8: For metric-based robustness in nonlinear updates. |
| Definition | 10 | 98 | Threshold auto-regressive herding model. | Mixing failure detection; bimodality tests. | Threshold-based majority-following for dependence. | Phase 7: Models herding in simulations. |
| Remark | 1 | 6 | Behavioral weights downweight noise traders. | Weight impact on posterior variance. | Reduces effective sample from noisy trades. | Phase 4: In behavioral weighting layer. |
| Remark | 2 | 11 | Hybrid prior balances structural and ML priors. | Prior strength sensitivity. | Avoids over-reliance on one prior type. | Phase 3: Hybrid prior setup. |
| Remark | 3 | 12 | Fractional Heston captures long memory in volatility. | Hurst parameter estimation. | Better fits empirical volatility persistence. | Phase 1: Volatility dynamics. |
| Remark | 4 | 14 | Nonlinear adjustments preserve tractability. | Computation time vs. accuracy. | Mixture of Betas remains closed-form. | Phase 6: Nonlinear corrections. |
| Remark | 5 | 21 | Robustness to small distortions. | Perturbation error bounds. | Posterior stable under small behavioral changes. | Phase 8: Robustness analysis. |
| Remark | 6 | 23 | Calibration of offsets via Empirical Bayes. | EB estimation convergence. | Fits offsets to historical markets. | Phase 6: Offset calibration. |
| Remark | 7 | 32 | Consistency under model assumptions. | Convergence rate in simulations. | Posterior converges to true p. | Phase 8: Asymptotics. |
| Remark | 8 | 33 | Role of effective sample size. | n* vs. n ratio. | Drives concentration, not raw trade count. | Phase 8: Finite-sample bounds. |
| Remark | 9 | 35 | Mixture for multimodal beliefs. | Multimodal detection. | Captures stratified data. | Phase 5: Multimodal posteriors. |
| Remark | 10 | 42 | Dynamic bias for time-varying distortions. | HMM fit quality. | Adapts to changing biases. | Phase 6: Dynamic layer. |
| Remark | 11 | 44 | Ising model for herding. | Mixing rate estimation. | Models dependence in trades. | Phase 7: Simulations. |
| Remark | 12 | 57 | Linear offsets as special case of nonlinear. | Reduction to linear model. | Recovers simple adjustments. | Phase 6: Structural corrections. |
| Remark | 13 | 57 | Representation of nonlinear distortions. | Nonlinear function approximation. | Approximates via finite regimes. | Phase 6: Nonlinear Stage 2. |
| Remark | 14 | 68 | Compatibility with mixtures and asymptotics. | Asymptotic behavior under mixtures. | Extends results to mixtures. | Phase 5 and 8: Mixtures and asymptotics. |
| Remark | 15 | 76 | Scope and limitations of projection. | Arbitrage constraint satisfaction. | Ensures no-arbitrage in higher dimensions. | Phase 7: Price projection. |
| Remark | 16 | 79 | Diagnosing mixing failure. | Autocorrelation analysis. | Detects strong dependence. | Phase 8: Dependence checks. |
| Remark | 17 | 81 | Role of large effective sample size. | n* growth rate. | Governs asymptotics. | Phase 8: Consistency. |
| Remark | 18 | 84 | Interpretation of error propagation. | Parameter error to posterior error. | Small estimation errors lead to small posterior distortions. | Phase 8: Error bounds. |
| Remark | 19 | 89 | Interpretation and choice of n*. | Effective sample calculation. | Adjusts for dependence. | Phase 8: Concentration. |
| Remark | 20 | 90 | Mixture extension and bootstrap refinements. | Bootstrap interval coverage. | Improves finite-n uncertainty. | Phase 7: Simulations. |
| Remark | 21 | 91 | Implications for Phase 7 metrics. | Coverage rates in simulations. | Targets finite-n errors. | Phase 7: Performance metrics. |
| Remark | 22 | 91 | Extension of Theorem 23. | Divergence bounds for mixtures. | Generalizes robustness. | Phase 8: Mixture robustness. |
| Remark | 23 | 93 | Small samples and long-shot regimes. | Skewness in low-n. | Caution in extremes. | Phase 8: Asymptotics limitations. |
| Remark | 24 | 95 | Practical implications in low-n and long-shot. | Concentration rate tests. | Uses Bernstein inequality. | Phase 8: Finite-sample. |
| Remark | 25 | 96 | Lipschitz regularization in calibration. | Regularization term impact. | Penalizes large Lipschitz constants. | Phase 6: Calibration. |
| Remark | 26 | 97 | Failure modes and relation to F2-F3. | Regime where robustness fails. | Limits of linear assumptions. | Phase 8: Nonlinear extensions. |
| Remark | 27 | 100 | Interpretation of impossibility. | Error lower bounds in strong herding. | Defines framework limits. | Phase 7: Simulations. |
| Remark | 28 | 102 | Practical guidance for long-shot markets. | Truncation effects on stability. | Stabilizes numerical issues. | Phase 8: Boundary behavior. |
| Remark | 29 | 104 | Interpretation and novelty of KL projection. | KL minimization verification. | Justifies Beta choice. | Phase 8: Information-theoretic view. |
| Remark | 30 | 104 | Limitations of KL projection view. | Approximation error in complex cases. | Trade-off in flexibility. | Phase 8: Alternatives. |
| Remark | 31 | 106 | Application to dynamic CAPOPM. | BvM error in HMM. | Handles mild nonstationarity. | Phase 6: Dynamic bias. |
| Remark | 32 | 108 | Interpretation for dynamic CAPOPM. | Feasibility in ergodic chains. | Positive and negative regimes. | Phase 6: Regime switching. |
| Theorem | 1 | 8 | Existence of unique equilibrium in parimutuel market. | Equilibrium computation convergence. | Unique price balancing supply and demand. | Phase 2: Posterior update. |
| Theorem | 2 | 10 | Consistency of structural prior. | Prior convergence rate. | Prior mean to true probability. | Phase 3: Hybrid prior. |
| Theorem | 3 | 21 | Hybrid prior remains robust under structural misspecification if bounded conditions on parameters hold, limiting error propagation from prior biases. | Prior mean bias under simulated misspecification; KL divergence from true prior. | Error in hybrid mean bounded by misspecification constant, with convergence as n increases. | Phase 2: Embedded in hybrid prior fusion to cap misspecification impacts during initial weighting. |
| Theorem | 4 | 22 | Hybridization projects ML uncertainty onto Beta via moment-matching, preserving distributional features like variance. | Wasserstein distance between projected and original ML distribution; variance preservation ratio. | Projected Beta matches ML mean/variance within epsilon, maintaining conjugacy. | Phase 2: Used in ML prior conversion to Beta for seamless integration into conjugate updates. |
| Theorem | 5 | 23 | Divergence bounds hold for hybrid priors under mismatch on strike grids, ensuring robustness across maturities. | Total variation distance on grid points; sensitivity to strike perturbations. | Supremum divergence over grid decays with sample size, under Lipschitz assumptions. | Phase 3: Applied to prior mismatch checks in order flow aggregation for multi-strike pricing. |
| Theorem | 6 | 31 | Bayesian-Nash equilibrium exists with signal-driven strategies in small-trader parimutuel markets, yielding informative odds. | Equilibrium convergence iterations; information entropy of induced odds. | Unique equilibrium where strategies rationalize observed order flow under risk neutrality. | Phase 3: Computes equilibrium odds in trader simulation for generating synthetic tapes. |
| Theorem | 7 | 32 | Pooling/non-pooling equilibria lead to asymptotic unraveling as trader count N grows, revealing true probabilities. | Unraveling rate vs. N; bias in empirical frequencies. | Frequencies converge to true p as N→∞, with pooling vanishing. | Phase 3: Simulates large-N unraveling in market simulator to validate info efficiency. |
| Theorem | 8 | 34 | Approximate conditional independence holds in large markets, justifying i.i.d. likelihood approximations. | Dependence measure (e.g., covariance of strategies); approximation error. | Covariance decays as 1/N, enabling Beta-Binomial treatment. | Phase 4: Justifies i.i.d. assumption in likelihood computation from trade tapes. |
| Theorem | 9 | 37 | Posterior predictive distribution is Beta-Binomial marginal, enabling closed-form pricing. | Predictive log-likelihood on holdout data; moment matching. | Marginal matches exact Beta-Binomial under conjugacy. | Phase 4: Computes predictive means for digital payoffs in posterior pipeline. |
| Theorem | 10 | 41 | CAPOPM implies risk-neutral CDF, density, and pricing kernel from posterior, compatible with Heston family. | Kernel shape plots; martingale verification. | Induced kernel positive and integrates to 1, matching Heston tails. | Phase 5: Generates risk-neutral densities for derivative pricing outputs. |
| Theorem | 11 | 43 | CAPOPM kernels exhibit shapes addressing asset pricing puzzles, like volatility smirk. | Kernel asymmetry metrics; puzzle resolution scores (e.g., slope). | Kernel monotonic in strikes, capturing empirical skew. | Phase 5: Fits kernels to simulated data for puzzle diagnostics. |
| Theorem | 12 | 46 | Kernel regularization preserves no-arbitrage in CAPOPM posteriors via smoothing. | Arbitrage violation counts pre/post-regularization; positivity checks. | Regularized kernel martingale and positive without violations. | Phase 7: Applies smoothing in pricing module to enforce arbitrage-freeness. |
| Theorem | 13 | 48 | Posterior-predictive fair prices for YES/NO contracts are arbitrage-free under risk-neutral valuation. | Price sum and monotonicity tests; super/sub-replication bounds. | Prices sum to 1, non-negative, and monotone in beliefs. | Phase 7: Finalizes YES/NO prices in calibration layer. |
| Theorem | 14 | 51 | No unimodal Beta uniformly approximates strongly multimodal mixtures, leading to divergence lower bounds. | TV distance from mixture to best Beta; mode separation tests. | Divergence bounded below by mode gap, preventing uniform fit. | Phase 5: Triggers mixture mode in multimodal detection for approximations. |
| Theorem | 15 | 56 | Mixture-of-Beta conjugacy under nonlinear structural corrections. | Mixture log-likelihood; Beta fit. | Yields finite mixture of Betas. | Phase 5: Multimodal posteriors. |
| Theorem | 16 | 61 | Lipschitz robustness of adjusted price. | Perturbation bounds on price. | Small changes in counts produce small price changes. | Phase 6: Bias correction. |
| Theorem | 17 | 77 | Arbitrage-free CAPOPM YES/NO prices. | No-arbitrage checks; Lipschitz continuity. | Enforces arbitrage-free prices. | Phase 7: Price calibration. |
| Theorem | 18 | 78 | Mixing regimes for Ising-type herding. | Alpha-mixing coefficients. | Geometric mixing in weak-coupling. | Phase 7: Herding simulations. |
| Theorem | 19 | 81 | Posterior consistency under model assumptions. | Convergence to true p in probability. | Posterior concentrates at true value. | Phase 8: Asymptotics. |
| Theorem | 20 | 82 | Consistency under behavioral distortion. | Consistency post-correction. | Remains consistent after Stage 1. | Phase 4: Behavioral weighting. |
| Theorem | 21 | 82 | Adversarial robustness. | Consistency under contamination <1/2. | Posterior consistent with adversaries. | Phase 8: Robustness. |
| Theorem | 22 | 83 | Lipschitz error propagation for CAPOPM posterior. | Error bounds in mean/variance/distribution. | Small parameter errors to small posterior errors. | Phase 8: Error propagation. |
| Theorem | 23 | 85 | Offset robustness. | Asymptotics with bounded offsets. | Consistency with O(1) offsets. | Phase 6: Structural offsets. |
| Theorem | 24 | 85 | Arbitrage-freeness of CAPOPM pricing. | Price sum =1, positivity, monotonicity. | Preserves no-arbitrage. | Phase 7: Pricing. |
| Theorem | 25 | 86 | Asymptotic distribution (Bernstein-von Mises). | Normality of posterior; variance match. | Posterior normal with 1/n rate. | Phase 8: Asymptotics. |
| Theorem | 26 | 88 | Finite-sample concentration for posterior mean. | Sub-Gaussian bound verification. | Concentrates at sub-Gaussian rate. | Phase 8: Finite-sample. |
| Theorem | 27 | 89 | Berry-Esseen-type bound for Beta posterior. | Supremum over CDF difference. | Finite-n normal approximation error O(1/sqrt(n)). | Phase 8: Asymptotics. |
| Theorem | 28 | 90 | Mixture posterior robustness bound. | TV distance bound. | Inherits component robustness. | Phase 5: Mixtures. |
| Theorem | 29 | 90 | Lower bound: mixture divergence cannot vanish under mode separation. | Divergence lower bound. | Unimodal approximations fail. | Phase 5: Mixtures. |
| Theorem | 30 | 93 | Consistency and normality of CAPOPM posterior under dependence. | Convergence and BvM. | Asymptotics under mixing. | Phase 8: Dependence. |
| Theorem | 31 | 96 | Wasserstein robustness of nonlinear posterior updates. | W1 bound. | Lipschitz in data metric. | Phase 8: Nonlinear. |
| Theorem | 32 | 97 | Hellinger stability under smooth monotone transformations. | Hellinger bound. | Stability under smooth maps. | Phase 8: Nonlinear. |
| Theorem | 33 | 99 | Impossibility of uniform consistency under strong herding. | Error lower bounds. | Fails in strong herding. | Phase 7: Herding. |
| Theorem | 34 | 104 | KL projection theorem for CAPOPM. | KL minimization. | Beta as KL projection. | Phase 8: Information-theoretic. |
| Theorem | 35 | 106 | Bernstein-von Mises under ergodic regime switching. | Asymptotic normality. | Handles mild nonstationarity. | Phase 6: Switching. |
| Theorem | 36 | 108 | Impossibility of uniform consistency under fast regime switching. | No convergence. | Fails in fast switching. | Phase 6: Switching. |
| Proposition | 1 | 8 | Robustness of posterior to small perturbations. | Perturbation error. | Lipschitz bound on mean. | Phase 4: Update. |
| Proposition | 2 | 13 | Martingale property in idealized case. | Conditional expectation check. | Posterior mean is martingale. | Phase 4: Sequential update. |
| Proposition | 3 | 17 | Monotonicity: Improving ANN/RNN metrics increases ML reliability weight r in hybrid. | Sensitivity of r to metric changes; correlation with accuracy. | r strictly increases with any positive metric shift, holding others fixed. | Phase 2: Updates r dynamically in ML prior weighting during fusion. |
| Proposition | 4 | 20 | Hybrid prior bias decomposes into structural and ML components, weighted by strengths. | Bias decomposition residuals; attribution analysis. | Total bias equals weighted sum of component biases. | Phase 2: Computes bias terms for diagnostic reporting in prior setup. |
| Proposition | 3.1 | 29 | Monotone likelihood ratio holds for order flow under signal quality ordering. | Likelihood ratio increments; MLR family tests. | Ratio strictly increasing in YES counts, implying ordered family. | Phase 3: Validates MLR in likelihood ratio computations for informativeness. |
| Proposition | 5 | 37 | Posterior moments (mean/variance) are explicit rationals of Beta parameters. | Moment computation accuracy; variance bounds. | Mean = α_post / (α_post + β_post), variance as standard Beta formula. | Phase 4: Extracts moments for variance-aware pricing in updates. |
| Proposition | 6 | 47 | Posterior predictive for digital payoffs is Bernoulli with Beta-marginalized mean. | Predictive CDF matching; Brier score on synthetics. | P(Z=1) = posterior mean, with variance preserved. | Phase 4: Generates payoff distributions for contract valuation. |
| Proposition | 7 | 48 | Posterior price strictly increasing in YES votes y. | Partial derivative signs; monotonicity plots. | dπ/dy > 0 for fixed n. | Phase 4: Ensures intuitive vote-price links in update rules. |
| Proposition | 8 | 48 | Posterior price continuous in counts y, n and parameters α0, β0. | Continuity checks via limits; epsilon-delta bounds. | Small changes in inputs yield small price changes. | Phase 4: Supports numerical stability in sequential updates. |
| Proposition | 9 | 49 | Mixture posterior mean and predictive are weighted averages of component Betas. | Mean computation under weights; predictive KL. | Mean = Σ w_k * μ_k, predictive matches mixture marginal. | Phase 5: Aggregates mixture moments for efficient pricing. |
| Proposition | 10 | 59 | Adjusted posterior mean/variance/distribution robust to pseudo-count perturbations. | Sensitivity to Δy*, Δn*; distribution distances. | Changes bounded by perturbation norms. | Phase 6: Applies in bias correction for perturbation diagnostics. |
| Proposition | 11 | 61 | No-arbitrage identity preserved post-adjustment: adjusted YES + NO = 1. | Sum checks across offsets; invariance tests. | π_adj_YES + π_adj_NO = 1 for admissible offsets. | Phase 6: Verifies identity in correction layer outputs. |
| Proposition | 12 | 63 | Posterior mean is a martingale in idealized sequential updates without biases. | Conditional expectation equality; martingale variance. | E[π_{t+1}|F_t] = π_t. | Phase 4: Sequential martingale property. |
| Proposition | 13 | 73 | Geometric alpha-mixing in high-temperature herding regime on bounded graphs. | Mixing coefficient decay; dependence tests. | α(k) ≤ ρ^k for ρ < 1, uniformly bounded degrees. | Phase 7: Justifies mixing in herding simulations for asymptotics. |
| Proposition | 14 | 74 | CAPOPM mapping continuous in priors and correction rules. | Lipschitz constant of full map; continuity plots. | Small input changes yield small output prices. | Phase 7: Ensures stability in end-to-end pipeline evaluations. |
| Proposition | 15 | 82 | Structural prior dominance: if q_str = p_true, posterior consistent regardless of ML. | Convergence under ML noise; bias vanishing. | Posterior → p_true as n* → ∞, ML irrelevant. | Phase 8: Tests dominance in asymptotic simulations. |
| Proposition | 16 | 82 | ML prior dominance: if p_ML = p_true and n_ML → ∞, posterior consistent despite structural errors. | Similar to Prop 15, swapping roles. | Posterior → p_true, structural negligible. | Phase 8: Validates in misspecification grids. |
| Proposition | 17 | 85 | Lipschitz stability of posterior mean to perturbations in adjustments. | Mean deviation bounds; Lipschitz verification. | |ˆp_adj - ˆp| bounded. | Phase 8: Stability checks. |
| Proposition | 18 | 94 | Finite-sample concentration under alpha-mixing for adjusted YES process. | Concentration inequalities; tail probabilities. | P(|¯Z - p| > ε) ≤ exp(-c n*). | Phase 8: Finite-sample bounds. |
| Proposition | 19 | 102 | Stabilized posterior via truncation or logit transform. | Stability and moment existence. | Improves finite-n behavior. | Phase 8: Boundary stabilization. |

---

## Section 3: Interconnections Table

This table summarizes dependencies and supports between elements for pipeline integrity.

| Element | Depends On | Supports | Key Metric Link | Phase Impact |
|---------|------------|----------|-----------------|--------------|
| Thm 3 | Assum 1 (params) | Thm 22 (error prop) | KL divergence | Prior fusion stability |
| Thm 6 | Assum A1-A6 | Prop 3.1 (MLR) | Equilibrium entropy | Trader simulation |
| Thm 12 | Thm 10 (CDF) | Prop 11 (no-arb) | Violation counts | Kernel smoothing |
| Prop 9 | Def 2 (mixture) | Thm 14 (unimodal fail) | Weighted mean error | Multimodal aggregation |
| Prop 18 | Prop 13 (mixing) | Thm 30 (normality) | Tail prob bounds | Asymptotic validation |

---

**END OF APPENDIX**