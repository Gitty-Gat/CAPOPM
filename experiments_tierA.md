# Tier A — Mechanism Validation Experiments (A1–A3)

## A1. INFO_EFFICIENCY_CURVES

1. **Experiment ID and Name**  
   A1.INFO_EFFICIENCY_CURVES — Information aggregation efficiency curves.

2. **Tier and Category**  
   Tier A — Mechanism Validation.

3. **Theoretical Claim Tested**  
   - Paper refs: Phase 4 (Bayesian update), Proposition 6, Lemma 3.  
   - Falsifiable claim: As informed-signal quality and informed participation rise (holding structural prior fixed), CAPOPM posteriors approach \(p_{\text{true}}\) faster than raw parimutuel implied probabilities and do so monotonically along signal quality.

4. **Experimental Design**  
   - Controlled variables: structural prior parameters (Phase 1), market window length, trade arrival process form.  
   - Swept parameters: informed trader share, informed signal accuracy, adversarial trader share; optional sweep of pool seeding strength.  
   - Fixed parameters: base liquidity schedule, correction strengths (Stage 1/2) per default config, baseline models list.  
   - Randomness and seed policy: deterministic global seed per scenario; all RNG streams derived from that seed; paired seeds across models for metric comparability.

5. **Simulation Procedure**  
   - Step 1: Instantiate scenario with specified informed/adversarial shares and signal accuracies.  
   - Step 2: Simulate transaction-level trades over the market window with dynamic parimutuel pool updates.  
   - Step 3: Run CAPOPM pipeline (Phases 1–6) to produce posterior paths; collect raw parimutuel implied probabilities as baseline.  
   - Step 4: Record posterior means and credible intervals at each timestep; mark final (market-close) posterior.  
   - Step 5: Compute metrics using paired seeds; aggregate across sweeps. Measurements taken after Phase 6 outputs and baseline runs.

6. **Required Metrics**  
   - Proper scoring: Brier score, Log score (final and time-path averages).  
   - Calibration: ECE, reliability tables.  
   - Bias/consistency: Posterior mean bias, MAE vs \(p_{\text{true}}\).  
   - Uncertainty: Coverage (90%, 95%), posterior variance ratio vs baseline.  
   - Geometry: Wasserstein distance in Beta space (CAPOPM vs truth-induced Beta).  
   - Learning curves: error vs timestep (uses existing metrics; no new metric).

7. **Required Outputs / Artifacts**  
   - `metrics_aggregated.csv` with columns including scenario_name, experiment_id, tier, model, seed, sweep parameters.  
   - `reliability_<model>.csv` for each model.  
   - `tests.csv` capturing paired statistical tests (paired t, Wilcoxon, bootstrap CI) across sweeps.  
   - `summary.json` summarizing pass/fail and sweep-wide tendencies.  
   - Optional plots: learning curves per sweep setting (stored under results/<scenario_name>/figures/).

8. **Pass / Fail Criteria**  
   - Monotonic improvement: Brier and Log scores improve (decrease/increase respectively) as signal quality increases for CAPOPM.  
   - Dominance: CAPOPM final Brier/log strictly better than raw parimutuel in ≥90% of paired seeds; no worse by more than tolerance ε=1e-3 in remaining cases.  
   - Calibration: ECE not worse than baseline; coverage within ±5% of nominal.  
   - Failure if monotonicity breaks or dominance fails under majority of sweeps.

9. **Dependencies**  
   - Requires validated Phase 1–6 outputs (core pipeline locked).  
   - Baselines already implemented; no new logic.  
   - Standalone within Tier A; prerequisite for Tier B claims using learning behavior.

10. **Failure Signatures**  
   - Learning curves flatten or reverse with higher signal quality.  
   - CAPOPM worse than parimutuel in final Brier/log.  
   - Overconfident intervals (coverage below 85%) or underconfident (>> nominal).  
   - Reliability plots show systematic bias (e.g., long-shot bias persisting).

---

## A2. TIME_TO_CONVERGE

1. **Experiment ID and Name**  
   A2.TIME_TO_CONVERGE — Convergence speed vs liquidity.

2. **Tier and Category**  
   Tier A — Mechanism Validation.

3. **Theoretical Claim Tested**  
   - Paper refs: Phase 4 consistency arguments; implicit convergence properties underlying Proposition 6.  
   - Falsifiable claim: Increasing liquidity (higher arrival intensity and pool seeding) reduces time-to-target-error for CAPOPM posteriors more than for uncorrected baselines, without increasing terminal bias.

4. **Experimental Design**  
   - Controlled variables: structural prior, trader type mix (shares fixed), signal quality fixed at mid-level.  
   - Swept parameters: liquidity level (arrival rate), pool seeding strength, number of steps.  
   - Fixed parameters: correction strengths, adversarial share held constant.  
   - Randomness and seed policy: deterministic global seed per liquidity setting; shared seeds across models; repeated runs per setting for variance estimates.

5. **Simulation Procedure**  
   - Step 1: Configure scenario with fixed trader mix and signal quality; vary liquidity and seeding per sweep grid.  
   - Step 2: Simulate transaction streams; update pools dynamically.  
   - Step 3: Run CAPOPM pipeline to obtain posterior over time; collect uncorrected baselines.  
   - Step 4: For each run, compute error vs time and identify timestep where error crosses predefined thresholds.  
   - Step 5: Aggregate time-to-threshold and variance decay metrics across seeds. Measurements taken after Phase 6 outputs at each timestep.

6. **Required Metrics**  
   - Time-to-ε error (first timestep where Brier ≤ ε; use ε grid, e.g., {0.05, 0.1}).  
   - Variance decay rate (slope of posterior variance vs time).  
   - Final Brier/Log for dominance checks.  
   - Coverage at market close.  
   - All metrics use existing implementations; no NEW metric types.

7. **Required Outputs / Artifacts**  
   - `metrics_aggregated.csv` with time-to-ε fields and variance decay summaries, plus required metadata columns.  
   - `reliability_<model>.csv` (final window).  
   - `tests.csv` for paired comparisons of time-to-ε and variance decay slopes.  
   - `summary.json` capturing convergence advantages and any regressions.  
   - Optional plots: time-to-ε curves and variance trajectories per liquidity level.

8. **Pass / Fail Criteria**  
   - CAPOPM time-to-ε strictly less than uncorrected baseline in ≥80% of paired seeds across liquidity levels.  
   - Variance decay rate magnitude greater (faster) for CAPOPM without final-score degradation (Brier/log not worse by >1e-3).  
   - Fail if higher liquidity does not shorten convergence or if corrections slow convergence materially.

9. **Dependencies**  
   - Requires successful A1 (information efficiency) to ensure microstructure sanity before convergence claims.  
   - Uses existing pipeline; no new logic.

10. **Failure Signatures**  
   - Time-to-ε increases with liquidity or is flat.  
   - CAPOPM slower than uncorrected baseline.  
   - Variance decay stalls; credible intervals remain wide.  
   - Final calibration degrades when liquidity increases (overconfidence).

---

## A3. STRATEGIC_TIMING_ATTACK

1. **Experiment ID and Name**  
   A3.STRATEGIC_TIMING_ATTACK — Late-manipulation robustness.

2. **Tier and Category**  
   Tier A — Mechanism Validation.

3. **Theoretical Claim Tested**  
   - Paper refs: Theorem 12; Proposition 9; Phase 6 Stage 1/2 corrections.  
   - Falsifiable claim: CAPOPM with Stage 1 + Stage 2 corrections exhibits reduced final-price bias and regret under concentrated late adversarial trades compared to uncorrected baselines and Stage-1-only variants.

4. **Experimental Design**  
   - Controlled variables: structural prior, informed trader signal quality, total market length.  
   - Swept parameters: fraction of adversarial volume shifted to final k timesteps; adversarial order size scaling; presence/absence of Stage 2 mixture.  
   - Fixed parameters: baseline informed/adversarial shares, correction hyperparameters at default.  
   - Randomness and seed policy: deterministic global seed; paired seeds across models and attack strengths; repeated draws for robustness.

5. **Simulation Procedure**  
   - Step 1: Define attack schedule allocating adversarial trades to late window; set scaling factors.  
   - Step 2: Simulate transaction-level stream with dynamic pools; apply attack timing.  
   - Step 3: Run CAPOPM with full corrections; run Stage-1-only and uncorrected baselines for comparison.  
   - Step 4: Capture posterior paths and final posteriors; log trade-level effective counts after Stage 1 and regime weights after Stage 2.  
   - Step 5: Compute bias and regret metrics; aggregate by attack strength. Measurements after Phase 6 outputs and baseline runs.

6. **Required Metrics**  
   - Final Brier and Log scores.  
   - Bias of final posterior mean vs \(p_{\text{true}}\).  
   - Regret vs uncorrected baseline (log-score regret).  
   - Coverage at 90%, 95%.  
   - Regime weight entropy and max weight (uses existing regime diagnostics).  
   - No new metrics required.

7. **Required Outputs / Artifacts**  
   - `metrics_aggregated.csv` with attack parameters, required metadata columns, bias/regret fields.  
   - `reliability_<model>.csv` (final window).  
   - `tests.csv` with paired tests comparing CAPOPM vs baselines under each attack strength.  
   - `summary.json` noting robustness outcomes and thresholds crossed.  
   - Optional plots: bias vs attack strength; regime weight trajectories.

8. **Pass / Fail Criteria**  
   - CAPOPM final bias magnitude ≤ 50% of uncorrected baseline in ≥80% of attack settings.  
   - Regret: expected log-score regret relative to truth lower than uncorrected and Stage-1-only in paired tests (p<0.05 after Holm correction).  
   - Coverage remains within ±5% of nominal; failure if overconfidence emerges under attacks.

9. **Dependencies**  
   - Requires Stage 1 and Stage 2 corrections operational (core pipeline).  
   - Depends on A1 (info efficiency) to ensure base behavior; independent of A2 except for reuse of time-path instrumentation.

10. **Failure Signatures**  
   - Final prices drift toward adversary despite corrections.  
   - Regime mixture collapses to incorrect regime or stays uniform with strong evidence.  
   - Bias/regret equal or worse than uncorrected; coverage drops sharply under attack.  
   - Reliability curves show late-window miscalibration spikes.

---

## Tier A Completion Checklist
- [ ] A1.INFO_EFFICIENCY_CURVES
- [ ] A2.TIME_TO_CONVERGE
- [ ] A3.STRATEGIC_TIMING_ATTACK
