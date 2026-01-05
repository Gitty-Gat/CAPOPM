# STAGE_B1_AST_SANITY_CHECK.md
## Purpose
Define an AST-based static enforcement gate for Stage B.1 that prevents:
1) Forbidden runtime gating (e.g., invariant_id="M-3")
2) Forbidden dominance/regret/oracle gating proxies
3) A2 threshold-style criteria on var_decay_slope (A2 is descriptive-only)
4) Forbidden metric keys in executable artifacts
5) Forbidden overclaim phrases in executable output fields

This gate is fail-closed: any violation aborts before experiments run.

## Inputs
- Policy file: repo_root/forbidden_policy.txt
- Scan targets:
  - src/**
  - scripts/**
  - src/capopm/experiments/**
  - smoke_test_*.py (repo root)
- Skip any file whose relative path matches ALLOW_PATHS patterns in forbidden_policy.txt
- Only parse .py files

## Output (on violation)
For each violation, print:
- file path
- line:col
- rule_id (from forbidden_policy.txt)
- short description
Exit code must be nonzero if any violation exists.

## Policy Rule Types (AST Semantics)

### 1) CALL_KW_LITERAL
Detect calls to a specific function where a keyword arg equals a specific literal string.

AST matching:
- Node type: ast.Call
- Function name match:
  - ast.Name(id=func) OR ast.Attribute(attr=func)
- Keyword match:
  - kw.arg == target kw
  - kw.value is ast.Constant(str)
- If kw.value == value => violation

Example:
- require_invariant(..., invariant_id="M-3") => violation

### 2) CALL_KW_CONTAINS_ANY
Detect calls where keyword arg is a literal string containing any forbidden token.

Same as CALL_KW_LITERAL, except:
- violation if any token is a substring of the literal string

### 3) CALL_KW_MUST_BE_LITERAL
Detect calls where keyword arg exists but is NOT a literal string.
- If kw.value is not ast.Constant(str) => violation
Rationale: dynamic invariant IDs defeat auditability.

### 4) DICT_LITERAL_KEY_FORBID
Detect dict literals containing forbidden keys.
- Node type: ast.Dict
- If any key is ast.Constant(str) in forbidden key set => violation

### 5) SUBSCRIPT_LITERAL_KEY_FORBID
Detect x["forbidden_key"] access/assignment with literal keys.
- Node type: ast.Subscript
- If subscript slice is ast.Constant(str) in forbidden key set => violation

### 6) DICT_LITERAL_KEYS_COEXIST
Detect dict literals containing ALL required keys (literal string keys).
- Node type: ast.Dict
- Build set of literal string keys
- If required_keys ⊆ present_keys => violation

Used to enforce:
- var_decay_slope + threshold (forbidden)
- var_decay_slope + direction (forbidden)

### 7) OUTPUT_DICT_VALUE_FORBID_PHRASE
Detect forbidden phrases in output-field values when they appear inside dict literals.

- Node type: ast.Dict
- For each (key, value) pair:
  - if key is ast.Constant(str) and key in OUTPUT_FIELDS
  - and value is ast.Constant(str) containing any forbidden phrase
  => violation

Important: do NOT scan all string constants, to avoid tripping on docstrings.

## Algorithm (reference implementation)
1) Parse forbidden_policy.txt:
   - Read ALLOW_PATHS patterns
   - Load rule blocks:
     - CALL_RULES
     - DICT_COEXIST_RULES
     - FORBIDDEN_METRIC_KEYS
     - OUTPUT_FIELDS
     - FORBIDDEN_PHRASES
     - OUTPUT_TEXT_RULES
2) Enumerate scan files:
   - Collect .py files in scan targets
   - Exclude allowlisted paths
3) For each file:
   - ast.parse(file_text)
   - Walk AST nodes with ast.NodeVisitor
   - Evaluate applicable rules
   - Record violations
4) Print violations, exit(1) if any, else exit(0)

## Acceptance Tests (must pass)
1) If runner.py contains require_invariant(invariant_id="M-3") => checker fails with rule_id M3_GATE
2) If code constructs {"dominance": ...} => checker fails with FORBIDDEN_METRIC_KEY_DICT
3) If code contains criteria dict {"var_decay_slope":..., "threshold":...} => checker fails with A2_SLOPE_THRESHOLD_FORBIDDEN
4) If a summary dict sets {"verdict":"entropy goes to zero"} => checker fails with FORBIDDEN_OUTPUT_PHRASE
5) Docstrings mentioning "dominates" do NOT fail (unless placed as values in output fields)

## Known Limitations
- Dynamic dict keys (e.g., metrics[key] where key is variable) are not reliably detectable.
- Only literal keys/values are enforced; use code review for dynamic cases.
- Policy updates require updating forbidden_policy.txt (canonical) and rerunning checker.

## Change Control
Any modification to:
- forbidden_policy.txt grammar
- checker AST semantics
must be documented here with rationale and a test case.
