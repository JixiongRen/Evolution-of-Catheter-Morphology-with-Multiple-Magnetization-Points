"""Read-only contract for the n2 effective-subspace sensitivity candidate."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContractDecision:
    """Validation result for one requested interface use."""

    accepted: bool
    reasons: tuple[str, ...]


class N2EffectiveSubspaceInterfaceContract:
    """Guard the n2 effective-subspace sensitivity against unsupported use."""

    schema_version = "p2c150_n2_effective_subspace_v1"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.source_dir = (
            self.root
            / "experiment_records_ik_optimize/20260630110000_p2c149_n2_2d_effective_subspace_release_candidate"
        )
        self.summary = json.loads((self.source_dir / "summary.json").read_text(encoding="utf-8"))
        self.rows = self._read_rows(self.source_dir / "release_candidate_rows.csv")

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        """Read CSV rows as plain dictionaries."""
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def evidence_status(self) -> dict[str, Any]:
        """Return the evidence status used by the guard."""
        return {
            "source_experiment": self.summary["experiment_id"],
            "candidate_count": self.summary["candidate_count"],
            "effective_subspace_dimension": self.summary["effective_subspace_dimension"],
            "safe_radius_A": self.summary["safe_radius_A"],
            "local_2d_interface_candidate": self.summary["local_2d_interface_candidate"],
            "full_3d_position_publishable": self.summary["full_3d_position_publishable"],
            "outer_ik_allowed": self.summary["outer_ik_allowed"],
            "numeric_basis_available": False,
            "reason": self.summary["reason"],
        }

    def allowed_scope(self) -> str:
        """Return the allowed scope in human-readable form."""
        return "两点磁控、末端位置、同一物理解分支、小扰动半径内的二维有效子空间诊断"

    def forbidden_scope(self) -> str:
        """Return forbidden uses in human-readable form."""
        return "完整三维位置灵敏度、完整末端位姿灵敏度、跨分支复用、跨工作点复用、默认外层 IK、无基向量时生成电流步"

    def required_request(self) -> dict[str, Any]:
        """Return required fields for a valid read-only request."""
        return {
            "schema_version": self.schema_version,
            "magnet_count": 2,
            "terminal_quantity": "position",
            "task_representation": "effective_subspace",
            "requested_task_dimension_max": 2,
            "branch_conditioned": True,
            "current_step_radius_A_max": self.summary["safe_radius_A"],
            "requires_numeric_basis_for_step_generation": True,
        }

    def validate_request(self, request: dict[str, Any]) -> ContractDecision:
        """Validate whether a request stays inside the audited scope."""
        reasons: list[str] = []
        required = self.required_request()
        if request.get("schema_version") != required["schema_version"]:
            reasons.append("schema_version_mismatch")
        if request.get("magnet_count") != required["magnet_count"]:
            reasons.append("magnet_count_mismatch")
        if request.get("terminal_quantity") != required["terminal_quantity"]:
            reasons.append("terminal_quantity_mismatch")
        if request.get("task_representation") != required["task_representation"]:
            reasons.append("task_representation_mismatch")
        if int(request.get("requested_task_dimension", 99)) > required["requested_task_dimension_max"]:
            reasons.append("task_dimension_exceeds_effective_rank")
        if request.get("branch_conditioned") is not True:
            reasons.append("branch_condition_missing")
        if request.get("use_for_default_outer_ik") is True:
            reasons.append("default_outer_ik_forbidden")
        if request.get("request_full_3d_position") is True:
            reasons.append("full_3d_position_not_publishable")
        if request.get("request_full_pose") is True:
            reasons.append("full_pose_not_publishable")
        if request.get("generate_current_step") is True:
            reasons.append("numeric_basis_missing_for_step_generation")
        try:
            radius = float(request.get("current_step_radius_A", 0.0))
        except (TypeError, ValueError):
            radius = float("inf")
            reasons.append("current_step_radius_parse_failed")
        if radius > float(required["current_step_radius_A_max"]):
            reasons.append("current_step_radius_exceeded")
        return ContractDecision(accepted=not reasons, reasons=tuple(reasons))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe contract snapshot."""
        return {
            "schema_version": self.schema_version,
            "evidence_status": self.evidence_status(),
            "allowed_scope": self.allowed_scope(),
            "forbidden_scope": self.forbidden_scope(),
            "required_request": self.required_request(),
            "release_rows": self.rows,
        }
