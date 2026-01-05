"""
AST-based static enforcement gate for forbidden invariants, metrics, and phrases.

Policies are read from forbidden_policy.txt at repo root. Any violation causes a
non-zero exit code (fail-closed).
"""

from __future__ import annotations

import ast
import fnmatch
import sys
from pathlib import Path
from typing import Dict, List, Set


REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "forbidden_policy.txt"


class Policy:
    def __init__(self) -> None:
        self.allow_paths: List[str] = []
        self.forbidden_invariant_tokens: List[str] = []
        self.forbidden_metric_keys: Set[str] = set()
        self.forbidden_phrases: List[str] = []


def _parse_policy(path: Path) -> Policy:
    if not path.exists():
        print(f"ERROR: forbidden policy file missing: {path}")
        sys.exit(1)

    policy = Policy()
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if section == "[ALLOW_PATHS]":
            policy.allow_paths.append(line)
            continue
        if section == "[FORBIDDEN_METRIC_KEYS]" and line.startswith("key:"):
            key = line.split(":", 1)[1].strip().strip('"')
            policy.forbidden_metric_keys.add(key)
            continue
        if section == "[FORBIDDEN_TEXT_LITERALS]" and line.startswith("literal_contains:"):
            phrase = line.split(":", 1)[1].strip().strip('"')
            policy.forbidden_phrases.append(phrase)
            continue
        if section == "[FORBIDDEN_CALLS]":
            if "contains_any" in line:
                prefix = "contains_any:"
                try:
                    tokens_part = line.split(prefix, 1)[1]
                    tokens = ast.literal_eval(tokens_part)
                    policy.forbidden_invariant_tokens.extend([str(t) for t in tokens])
                except Exception:
                    policy.forbidden_invariant_tokens.extend(
                        ["dominance", "regret_dominance", "no_regret", "entropy_to_zero", "truth_recovery", "oracle"]
                    )
            continue
    if not policy.forbidden_invariant_tokens:
        policy.forbidden_invariant_tokens = [
            "dominance",
            "regret_dominance",
            "no_regret",
            "entropy_to_zero",
            "truth_recovery",
            "oracle",
        ]
    return policy


def _is_require_invariant(func: ast.AST) -> bool:
    return isinstance(func, ast.Name) and func.id == "require_invariant" or (
        isinstance(func, ast.Attribute) and func.attr == "require_invariant"
    )


def _key_from_slice(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Index) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):  # type: ignore[attr-defined]
        return node.value.value  # pragma: no cover
    return None


class ForbiddenVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, policy: Policy, violations: List[Dict], source: str) -> None:
        self.path = path
        self.policy = policy
        self.violations = violations
        self.source = source

    def visit_Call(self, node: ast.Call) -> None:
        if _is_require_invariant(node.func):
            inv_kw = next((kw for kw in node.keywords if kw.arg == "invariant_id"), None)
            if inv_kw is None or not isinstance(inv_kw.value, ast.Constant) or not isinstance(inv_kw.value.value, str):
                self._record(node, "FORBIDDEN_INVARIANT_ID", "require_invariant invariant_id must be literal string")
            else:
                inv_id = inv_kw.value.value
                if inv_id == "M-3":
                    self._record(node, "FORBIDDEN_INVARIANT_ID", 'require_invariant invariant_id="M-3" forbidden')
                for tok in self.policy.forbidden_invariant_tokens:
                    if tok and tok in inv_id:
                        self._record(node, "FORBIDDEN_INVARIANT_TOKEN", f"forbidden invariant token in id: {inv_id}")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        keys = []
        for k_node, v_node in zip(node.keys, node.values):
            if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                key = k_node.value
                keys.append(key)
                if key in self.policy.forbidden_metric_keys:
                    self._record(k_node, "FORBIDDEN_METRIC_KEY", f"forbidden key '{key}' in dict literal")
                if key in {"status", "verdict", "conclusion", "claim", "result", "audit_verdict"}:
                    if isinstance(v_node, ast.Constant) and isinstance(v_node.value, str):
                        for phrase in self.policy.forbidden_phrases:
                            if phrase and phrase.lower() in v_node.value.lower():
                                self._record(v_node, "FORBIDDEN_PHRASE", f"forbidden phrase in output: '{phrase}'")
        if "var_decay_slope" in keys and ("threshold" in keys or "direction" in keys):
            self._record(node, "A2_SLOPE_THRESHOLD_FORBIDDEN", "var_decay_slope with threshold/direction is forbidden")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = _key_from_slice(node.slice)  # type: ignore[arg-type]
        if key and key in self.policy.forbidden_metric_keys:
            self._record(node, "FORBIDDEN_METRIC_KEY", f"forbidden metric key access '{key}'")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if self._is_loss_gate(node.test):
            self._record(
                node,
                "FORBIDDEN_M3_GATING_CONSTRUCT",
                "Assert-based gating on capopm vs baseline loss comparison is forbidden",
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self._is_loss_gate(node.test) and self._body_has_abort(node.body):
            self._record(
                node,
                "FORBIDDEN_M3_GATING_CONSTRUCT",
                "If-based gating on capopm vs baseline loss comparison with abort action is forbidden",
            )
        self.generic_visit(node)

    def _record(self, node: ast.AST, category: str, message: str) -> None:
        self.violations.append(
            {
                "path": str(self.path),
                "line": getattr(node, "lineno", 0),
                "col": getattr(node, "col_offset", 0),
                "category": category,
                "message": message,
            }
        )

    def _is_loss_gate(self, test: ast.AST) -> bool:
        """Heuristic: test source mentions capopm vs baseline loss comparisons."""

        src = ast.get_source_segment(self.source, test) or ""
        src_lower = src.lower()
        has_loss = any(token in src_lower for token in ["brier", "log", "delta_brier", "delta_log"])
        has_capopm = "capopm" in src_lower
        has_baseline = any(tok in src_lower for tok in ["uncorrected", "baseline", "base"])
        return bool(src and has_loss and has_capopm and has_baseline)

    def _body_has_abort(self, body: List[ast.stmt]) -> bool:
        for stmt in body:
            if isinstance(stmt, (ast.Raise, ast.Assert)):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if self._is_exit_call(stmt.value):
                    return True
            # Recurse into nested blocks conservatively
            for child in ast.walk(stmt):
                if isinstance(child, (ast.Raise, ast.Assert)):
                    return True
                if isinstance(child, ast.Call) and self._is_exit_call(child):
                    return True
        return False

    def _is_exit_call(self, call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Name) and func.id in {"exit", "quit", "sys_exit"}:
            return True
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "sys" and func.attr in {"exit", "quit"}:
                return True
        return False


def _should_skip(path: Path, allow_patterns: List[str]) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in allow_patterns)


def _collect_files() -> List[Path]:
    targets = [REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "src" / "capopm" / "experiments"]
    files: Set[Path] = set()
    for target in targets:
        if target.exists():
            files.update(p for p in target.rglob("*.py") if p.is_file())
    files.update(p for p in REPO_ROOT.glob("smoke_test_*.py") if p.is_file())
    return sorted(files)


def main() -> None:
    policy = _parse_policy(POLICY_PATH)
    violations: List[Dict] = []
    for path in _collect_files():
        if _should_skip(path, policy.allow_paths):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError as exc:
            violations.append(
                {
                    "path": str(path),
                    "line": exc.lineno or 0,
                    "col": exc.offset or 0,
                    "category": "PARSE_ERROR",
                    "message": f"Syntax error: {exc}",
                }
            )
            continue
        ForbiddenVisitor(path, policy, violations, source).visit(tree)

    if violations:
        for v in violations:
            print(f"{v['path']}:{v['line']}:{v['col']}: [{v['category']}] {v['message']}")
        sys.exit(1)
    print("forbidden_ast_check: no violations detected.")


if __name__ == "__main__":
    main()
