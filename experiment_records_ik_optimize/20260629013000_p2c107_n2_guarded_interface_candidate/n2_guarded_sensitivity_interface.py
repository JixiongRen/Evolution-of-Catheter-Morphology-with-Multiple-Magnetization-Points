"""Guarded read-only interface for the n2 sensitivity candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReuseDecision:
    """Decision for one cache reuse request."""

    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GuardedEvidence:
    """Evidence summary for the guarded n2 sensitivity candidate."""

    task_name: str
    eps: float
    stationarity_norm: float
    force_ratio: float
    task_condition: float
    random_direction_passed: int
    random_direction_total: int
    random_direction_max_rel_err: float
    guarded_repeat_passed: int
    guarded_repeat_total: int
    guarded_repeat_max_rel_err: float
    guarded_repeat_same_status: bool
    status: str


class N2GuardedSensitivityInterface:
    """Expose the current n2 local sensitivity candidate with reuse guards."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.contract = self._read("experiment_records_ik_optimize/20260629003000_p2c105_n2_cache_contract_audit/cache_contract.json")
        self.reaudit = self._read("experiment_records_ik_optimize/20260629010000_p2c106_n2_contract_guarded_reaudit/summary.json")
        self.milestone = self._read("experiment_records_ik_optimize/20260628222500_p2c100_ik_sensitivity_milestone_synthesis/summary.json")

    def _read(self, rel_path: str) -> dict[str, Any]:
        return json.loads((self.root / rel_path).read_text(encoding="utf-8"))

    def validate_request(self, request: dict[str, Any]) -> ReuseDecision:
        """Validate a request against the recorded cache reuse contract."""
        reasons: list[str] = []
        if request.get("schema_version") != self.contract["schema_version"]:
            reasons.append("schema_version_mismatch")
        if request.get("task_name") != self.contract["task_name"]:
            reasons.append("task_name_mismatch")
        if abs(float(request.get("eps", float("nan"))) - float(self.contract["eps"])) > 1.0e-15:
            reasons.append("eps_mismatch")
        if request.get("branch_label") != "trusted_seed":
            reasons.append("branch_label_mismatch")
        if request.get("task_dimension") != 3:
            reasons.append("task_dimension_mismatch")
        if request.get("terminal_quantity") != "position":
            reasons.append("terminal_quantity_mismatch")
        return ReuseDecision(accepted=not reasons, reasons=tuple(reasons))

    def evidence(self) -> GuardedEvidence:
        """Return gate evidence for the guarded n2 position-only candidate."""
        gate = self.contract["gate_values"]
        repeat_rows = self.reaudit["repeat_rows"]
        max_repeat_rel = max(float(row["max_vector_rel_err"]) for row in repeat_rows)
        repeat_passed = sum(int(row["passed"]) for row in repeat_rows)
        repeat_total = sum(int(row["total"]) for row in repeat_rows)
        candidate_ok = (
            gate["stationarity_norm"] <= self.contract["gate_limits"]["stationarity_norm_max"]
            and gate["force_ratio"] <= self.contract["gate_limits"]["force_ratio_max"]
            and gate["task_condition"] <= self.contract["gate_limits"]["task_condition_max"]
            and gate["random_direction_max_rel_err"] <= self.contract["gate_limits"]["linearization_rel_err_max"]
            and gate["random_direction_passed"] == gate["random_direction_total"]
            and self.reaudit["contract_accepted"]
            and repeat_passed == repeat_total
            and max_repeat_rel <= self.contract["gate_limits"]["linearization_rel_err_max"]
            and self.reaudit["repeat_comparison"]["same_status_all"]
        )
        return GuardedEvidence(
            task_name="三维末端位置",
            eps=float(self.contract["eps"]),
            stationarity_norm=float(gate["stationarity_norm"]),
            force_ratio=float(gate["force_ratio"]),
            task_condition=float(gate["task_condition"]),
            random_direction_passed=int(gate["random_direction_passed"]),
            random_direction_total=int(gate["random_direction_total"]),
            random_direction_max_rel_err=float(gate["random_direction_max_rel_err"]),
            guarded_repeat_passed=repeat_passed,
            guarded_repeat_total=repeat_total,
            guarded_repeat_max_rel_err=max_repeat_rel,
            guarded_repeat_same_status=bool(self.reaudit["repeat_comparison"]["same_status_all"]),
            status="branch_conditioned_candidate" if candidate_ok else "diagnostic_only",
        )

    def can_use_for_default_outer_ik(self) -> bool:
        """Return whether this local derivative may be a default outer-IK model."""
        return False

    def allowed_scope(self) -> str:
        """Return the allowed engineering scope."""
        return "同一模型、同一目标点、同一基准电流、同一任务口径下的分支条件局部导数复核"

    def forbidden_scope(self) -> str:
        """Return the forbidden engineering scope."""
        return "全局灵敏度、完整末端位姿灵敏度、跨目标点复用、跨电流基点复用、默认外层 IK 模型"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe interface snapshot."""
        evidence = self.evidence()
        return {
            "evidence": evidence.__dict__,
            "allowed_scope": self.allowed_scope(),
            "forbidden_scope": self.forbidden_scope(),
            "outer_ik_default_allowed": self.can_use_for_default_outer_ik(),
            "required_request": {
                "schema_version": self.contract["schema_version"],
                "task_name": self.contract["task_name"],
                "eps": self.contract["eps"],
                "branch_label": "trusted_seed",
                "task_dimension": 3,
                "terminal_quantity": "position",
            },
        }
