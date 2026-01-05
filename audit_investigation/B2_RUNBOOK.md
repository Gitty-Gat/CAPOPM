CAPOPM B.2 runbook (execution only)

Commands
- Run all paper experiments with paper-ready runs: `python run_paper_suite.py --experiment ALL --runs 30`
- Run a single experiment (example B3): `python run_paper_suite.py --experiment B3 --runs 30`
- Override runs-per-cell (example 10): `python run_paper_suite.py --experiment ALL --runs 10`

Expected artifacts per scenario
- `results/<scenario>/summary.json`
- `results/<scenario>/audit.json` (when produced by audit runner)
- `results/<scenario>/metrics_aggregated.csv`
- `results/<scenario>/tests.csv`
- `results/<scenario>/reliability_*.csv`

Manifest checks
- `PAPER_RUN_MANIFEST.json` exists at repo root after suite runs.
- Manifest top-level fields include `created_utc`, `python_version`, `command`, `ast_gate`, `min_runs_per_cell`, `min_grid_points`.
- Each `paper_runs` entry includes `experiment`, `scenario_name`, `seed`, `base_seed`, `run_index`, `sweep_params`, `n_runs`, `results_dir`, `summary_path`, `audit_path`, `audit_hash`, `artifact_hashes`.

AST gate verification
- Run `python scripts/forbidden_ast_check.py` before executions; expect exit code 0.
