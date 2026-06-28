"""Guarded step-generation contract for n2 effective-subspace sensitivity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StepDecision:
    """Validation result for one 2D subspace step request."""

    accepted: bool
    reasons: tuple[str, ...]
    current_step_norm_A: float
    projected_task_norm_m: float


class N2TwoDimensionalStepInterfaceContract:
    """Generate guarded n2 local steps only inside the audited 2D subspace."""

    schema_version = "p2c153_n2_2d_step_v1"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.basis_path = (
            self.root
            / "experiment_records_ik_optimize/20260630120000_p2c151_n2_numeric_subspace_basis/numeric_subspace_basis.json"
        )
        self.closure_path = (
            self.root
            / "experiment_records_ik_optimize/20260630123000_p2c152_n2_2d_subspace_closure/summary.json"
        )
        self.basis = json.loads(self.basis_path.read_text(encoding="utf-8"))
        self.closure = json.loads(self.closure_path.read_text(encoding="utf-8"))
        self.records = {record["direction"]: record for record in self.basis["records"]}

    def allowed_workpoints(self) -> tuple[str, ...]:
        """Return audited workpoint names."""
        return tuple(sorted(self.records))

    def safe_radius_A(self) -> float:
        """Return the audited current-step radius."""
        return float(self.closure["safe_radius_A"])

    def can_generate_step(self) -> bool:
        """Return whether guarded 2D diagnostic step generation is allowed."""
        return bool(self.closure["all_passed"]) and bool(self.basis["summary"]["numeric_basis_available"])

    def allowed_scope(self) -> str:
        """Return the allowed scope in human-readable form."""
        return "两点磁控、末端位置、指定工作点、同一物理解分支、0.001 A 内二维有效子空间小步诊断"

    def forbidden_scope(self) -> str:
        """Return forbidden uses in human-readable form."""
        return "完整三维位置灵敏度、完整末端位姿灵敏度、跨工作点复用、跨分支复用、默认外层逆运动学、超过审计半径的线性外推"

    def _record(self, workpoint: str) -> dict[str, Any]:
        """Return one workpoint record."""
        if workpoint not in self.records:
            raise KeyError(f"unknown workpoint: {workpoint}")
        return self.records[workpoint]

    def project_task_delta(self, workpoint: str, dp_m: np.ndarray) -> np.ndarray:
        """Project a terminal-position increment into the audited 2D subspace."""
        record = self._record(workpoint)
        projector = np.asarray(record["task_projector_U2U2T"], dtype=float)
        return projector @ np.asarray(dp_m, dtype=float).reshape(3)

    def propose_current_step(self, workpoint: str, dp_m: np.ndarray) -> np.ndarray:
        """Compute the guarded rank-2 current step before radius validation."""
        record = self._record(workpoint)
        u2 = np.asarray(record["task_basis_U2"], dtype=float)
        v2 = np.asarray(record["current_basis_V2"], dtype=float)
        sigma = np.asarray(record["singular_values_m_per_A"][:2], dtype=float)
        dp2 = self.project_task_delta(workpoint, dp_m)
        coeff = u2.T @ dp2
        return v2.T @ (coeff / sigma)

    def validate_step_request(self, request: dict[str, Any]) -> StepDecision:
        """Validate a request and reject unsupported uses."""
        reasons: list[str] = []
        if request.get("schema_version") != self.schema_version:
            reasons.append("schema_version_mismatch")
        if request.get("magnet_count") != 2:
            reasons.append("magnet_count_mismatch")
        if request.get("terminal_quantity") != "position":
            reasons.append("terminal_quantity_mismatch")
        if request.get("task_representation") != "effective_subspace":
            reasons.append("task_representation_mismatch")
        if request.get("branch_conditioned") is not True:
            reasons.append("branch_condition_missing")
        if request.get("use_for_default_outer_ik") is True:
            reasons.append("default_outer_ik_forbidden")
        if request.get("request_full_3d_position") is True:
            reasons.append("full_3d_position_not_publishable")
        if request.get("request_full_pose") is True:
            reasons.append("full_pose_not_publishable")
        workpoint = str(request.get("workpoint", ""))
        if workpoint not in self.records:
            reasons.append("workpoint_not_audited")
        try:
            dp = np.asarray(request.get("desired_tip_delta_m"), dtype=float).reshape(3)
        except (TypeError, ValueError):
            dp = np.zeros(3, dtype=float)
            reasons.append("desired_tip_delta_parse_failed")
        if not self.can_generate_step():
            reasons.append("closure_evidence_missing")
        if workpoint in self.records:
            dI = self.propose_current_step(workpoint, dp)
            dp2 = self.project_task_delta(workpoint, dp)
        else:
            dI = np.zeros(8, dtype=float)
            dp2 = np.zeros(3, dtype=float)
        current_norm = float(np.linalg.norm(dI))
        projected_norm = float(np.linalg.norm(dp2))
        if current_norm > self.safe_radius_A() + 1.0e-15:
            reasons.append("current_step_radius_exceeded")
        return StepDecision(
            accepted=not reasons,
            reasons=tuple(reasons),
            current_step_norm_A=current_norm,
            projected_task_norm_m=projected_norm,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe contract snapshot."""
        return {
            "schema_version": self.schema_version,
            "allowed_scope": self.allowed_scope(),
            "forbidden_scope": self.forbidden_scope(),
            "allowed_workpoints": list(self.allowed_workpoints()),
            "safe_radius_A": self.safe_radius_A(),
            "can_generate_2d_diagnostic_step": self.can_generate_step(),
            "outer_ik_default_allowed": False,
            "closure_summary": self.closure,
            "basis_summary": self.basis["summary"],
            "required_request": {
                "schema_version": self.schema_version,
                "magnet_count": 2,
                "terminal_quantity": "position",
                "task_representation": "effective_subspace",
                "branch_conditioned": True,
                "workpoint": "one of allowed_workpoints",
                "desired_tip_delta_m": "3-vector; only its audited 2D projection is used",
                "use_for_default_outer_ik": False,
                "request_full_3d_position": False,
                "request_full_pose": False,
            },
        }
