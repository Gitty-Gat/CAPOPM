"""
Runtime invariant enforcement and observability logging for Stage B.1.

Enforces fail-closed semantics: any invariant violation raises
InvariantViolation immediately. Uses a context to record pass/fail status
and fallback activations without introducing new observables for claims.
B1-CHG-02/B1-CHG-03/B1-CHG-05 instrumentation anchor.
"""

from __future__ import annotations

import hashlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class InvariantViolation(RuntimeError):
    """Raised when a Stage B.1 invariant fails."""

    def __init__(self, invariant_id: str, message: str, data: Optional[Dict[str, Any]] = None):
        self.invariant_id = invariant_id
        self.data = data or {}
        super().__init__(f"[InvariantViolation:{invariant_id}] {message} | data={self.data}")


@dataclass
class InvariantRecord:
    invariant_id: str
    status: str  # "pass" or "fail"
    tolerance: Optional[float] = None
    detail: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackRecord:
    fallback_id: str
    detail: Dict[str, Any]


@dataclass
class InvariantContext:
    """Holds per-run invariant and fallback logs."""

    experiment_id: Optional[str]
    scenario_name: Optional[str]
    run_seed: Optional[int]
    config_hash: Optional[str]
    invariant_log: list = field(default_factory=list)
    fallback_log: list = field(default_factory=list)

    def record_invariant(self, rec: InvariantRecord) -> None:
        self.invariant_log.append(rec)

    def record_fallback(self, rec: FallbackRecord) -> None:
        self.fallback_log.append(rec)


_ctx: ContextVar[Optional[InvariantContext]] = ContextVar("invariant_ctx", default=None)


def set_invariant_context(ctx: Optional[InvariantContext]):
    return _ctx.set(ctx)


def reset_invariant_context(token) -> None:
    _ctx.reset(token)


def current_context() -> Optional[InvariantContext]:
    return _ctx.get()


def _build_data(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = current_context()
    data = dict(extra or {})
    if ctx:
        data.setdefault("experiment_id", ctx.experiment_id)
        data.setdefault("scenario_name", ctx.scenario_name)
        data.setdefault("run_seed", ctx.run_seed)
        data.setdefault("config_hash", ctx.config_hash)
    return data


def require_invariant(condition: bool, invariant_id: str, message: str, tolerance: Optional[float] = None, data: Optional[Dict[str, Any]] = None) -> None:
    """Assert an invariant, record status, and fail-closed on violation."""

    ctx = current_context()
    payload = _build_data(data)
    if condition:
        if ctx:
            ctx.record_invariant(
                InvariantRecord(
                    invariant_id=invariant_id,
                    status="pass",
                    tolerance=tolerance,
                    detail=message,
                    data=payload,
                )
            )
        return
    # Record failure and raise.
    if ctx:
        ctx.record_invariant(
            InvariantRecord(
                invariant_id=invariant_id,
                status="fail",
                tolerance=tolerance,
                detail=message,
                data=payload,
            )
        )
    raise InvariantViolation(invariant_id, message, data=payload)


def record_fallback(fallback_id: str, detail: Dict[str, Any]) -> None:
    """Log fallback/clamping/clipping activation."""

    ctx = current_context()
    if ctx:
        ctx.record_fallback(FallbackRecord(fallback_id=fallback_id, detail=_build_data(detail)))


def stable_config_hash(cfg: Dict) -> str:
    """Deterministic hash for config snapshots."""

    import json

    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
