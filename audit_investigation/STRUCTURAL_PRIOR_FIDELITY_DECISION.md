# Structural Prior Fidelity Decision (Stage B.1)

- **Mode selected:** `surrogate_heston` (standard Heston surrogate only; fractional Heston out-of-scope per STAGE_B0_GOVERNANCE_SPEC.md and STAGE_B1_OBSERVABILITY_CONTRACT_AND_INVARIANT_CHECKLIST.md).
- **Runtime enforcement:** `structural_prior_mode` is asserted in `posterior.capopm_pipeline` (invariant S-0). Any other mode aborts execution.
- **Invariants enforced:** S-1 (positivity), S-3 (tail monotonicity over K ∈ [0.5S0, 1.5S0], T=1, r=0, q=0) executed at runtime with fail-closed semantics. Violations raise `InvariantViolation`.
- **Logging:** `structural_prior_mode` and `config_hash` are included in summary metadata; invariant logs surface per-run and scenario-level in `summary.json` and per-run metrics.
- **Claim gating:** All claims that depend on Phase 1 fidelity remain surrogate-only. No theorem-level validation is asserted; this mode is a proxy for interface correctness and invariant enforcement. Fractional-dependent theorems remain untestable until a Phase-1-faithful implementation is provided.
- **Observability boundaries:** Structural parameter vector Θ remains latent (class L) and is not exposed to audits/claims. Only surrogate outputs q_sur(K,T) are observable (class S).
