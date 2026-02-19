# RESULTS_CITATION_INDEX

This index links empirical statements in the manuscript to concrete artifacts.

## Extraction Base

Primary artifact set:
- `results/**/audit.json`
- `results/**/summary.json`
- `results/**/tests.csv`

Scenario aggregation excludes utility folders:
- `results/paper_artifacts/`
- `results/mini_regression/`
- `results/smoke_metrics_phase7/`

## Claim-to-Artifact Mapping

| Empirical Statement | Manuscript Location | Artifact Path(s) | Extraction Method |
|---|---|---|---|
| Family-level determinate/indeterminate counts (`A1: 9/9 indeterminate`, `A2: 2 pass + 8 indeterminate`, `B5: 6 pass`, etc.) | `sections/09_validation_summary.tex` | `results/**/audit.json` | Parse `experiment_id` and `criteria_evaluation.overall_pass`; group by family prefix (`A1`..`B5`). |
| Family-level table in supplement (`Total Scenarios`, `Determinate Pass Count`, `Indeterminate Count`) | `supplement/s05_additional_tables.tex` | `results/**/audit.json` | Same aggregation as above using `criteria_evaluation.overall_pass`. |
| Statement that determinacy and paper-ready flags do not coincide uniformly | `sections/09_validation_summary.tex` | `results/**/audit.json` | Compare per-scenario `criteria_evaluation.overall_pass` vs `seed_grid_coverage.paper_ready`. |
| Statement that stress-sensitive families are not uniformly supported under current audits | `supplement/s02_stress_tests.tex` | `results/**/audit.json` | Filter B1/B4-related scenarios and observe predominately indeterminate `criteria_evaluation.overall_pass` plus frequent coverage flags. |
| Statement that low-support / coverage flags are widespread | `sections/10_discussion_limitations_future_work.tex`, `appendix/a03_extra_results.tex` | `results/**/audit.json` | Count scenarios with non-empty `borderline_flags.coverage_flags` and scenarios with any `coverage.*.extreme_p.warning_low_support == true`. |
| Scenario-level diagnostics are available for exact values | `appendix/a03_extra_results.tex` | `results/**/audit.json`, `results/**/summary.json`, `results/**/tests.csv` | Direct per-scenario lookup by folder name. |

## Reproducibility Commands Used During Refactor

### Family status aggregation
```powershell
$audits = Get-ChildItem results -Recurse -Filter audit.json |
  Where-Object { $_.FullName -notmatch '\\paper_artifacts\\' -and $_.FullName -notmatch '\\mini_regression\\' -and $_.FullName -notmatch '\\smoke_metrics_phase7\\' }
$rows = foreach($f in $audits){
  $j = Get-Content -Raw $f.FullName | ConvertFrom-Json
  [pscustomobject]@{
    Family = ($j.experiment_id -split '\\.')[0].ToUpper()
    OverallPass = $j.criteria_evaluation.overall_pass
    PaperReady = $j.seed_grid_coverage.paper_ready
  }
}
$rows | Group-Object Family
```

### Coverage / low-support prevalence
```powershell
$audits = Get-ChildItem results -Recurse -Filter audit.json |
  Where-Object { $_.FullName -notmatch '\\paper_artifacts\\' -and $_.FullName -notmatch '\\mini_regression\\' -and $_.FullName -notmatch '\\smoke_metrics_phase7\\' }
$rows = foreach($f in $audits){
  $j = Get-Content -Raw $f.FullName | ConvertFrom-Json
  [pscustomobject]@{
    CoverageFlags = ($j.borderline_flags.coverage_flags | Measure-Object).Count
    LowSupport = ($j.coverage.PSObject.Properties.Name | ForEach-Object { $j.coverage.$_.extreme_p.warning_low_support }) -contains $true
  }
}
$rows
```
