"""Candidate interface for n2 branch-conditioned sensitivity evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SensitivityDecision:
    """Decision record for one sensitivity task."""

    object_name: str
    task_name: str
    status: str
    allowed_use: str
    forbidden_use: str
    failure_or_risk: str


@dataclass(frozen=True)
class N2EvidenceSummary:
    """Certified evidence summary for the n2 trusted branch."""

    branch_label: str
    task_name: str
    eps_max: float
    target_radius_m: float
    random_current_passed: int
    random_current_total: int
    random_current_max_rel_err: float
    target_loop_passed: int
    target_loop_total: int
    status: str


class N2BranchConditionedSensitivityInterface:
    """Read-only interface for the current n2 sensitivity candidate."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.p2c97 = self._read("experiment_records_ik_optimize/20260628211500_p2c97_n2_publish_candidate_synthesis/summary.json")
        self.p2c98 = self._read("experiment_records_ik_optimize/20260628213500_p2c98_n2_eps0125_random_current_batch/summary.json")
        self.p2c100 = self._read("experiment_records_ik_optimize/20260628222500_p2c100_ik_sensitivity_milestone_synthesis/summary.json")

    def _read(self, rel_path: str) -> dict[str, Any]:
        return json.loads((self.root / rel_path).read_text(encoding="utf-8"))

    def decision(self) -> SensitivityDecision:
        """Return the engineering decision for n2 position-only sensitivity."""
        row = next(row for row in self.p2c100["rows"] if row["object"] == "n2 两点磁控" and row["task"] == "三维末端位置")
        return SensitivityDecision(
            object_name=row["object"],
            task_name=row["task"],
            status=row["decision"],
            allowed_use=row["allowed_use"],
            forbidden_use=row["forbidden_use"],
            failure_or_risk=row["failure_class"],
        )

    def evidence_summary(self) -> N2EvidenceSummary:
        """Return gate evidence for the n2 trusted-branch candidate."""
        random_rows = self.p2c98["audit_rows"]
        target_rows = [row for row in self.p2c97["rows"] if row["category"] == "目标闭环"]
        random_passed = sum(row["status"] == "publishable" for row in random_rows)
        target_passed = sum(row["status"] == "通过" for row in target_rows)
        return N2EvidenceSummary(
            branch_label="trusted_seed",
            task_name="三维末端位置",
            eps_max=float(self.p2c98["eps"]),
            target_radius_m=1.0e-4,
            random_current_passed=int(random_passed),
            random_current_total=len(random_rows),
            random_current_max_rel_err=max(float(row["vector_rel_err"]) for row in random_rows),
            target_loop_passed=int(target_passed),
            target_loop_total=len(target_rows),
            status="candidate" if random_passed == len(random_rows) and target_passed == len(target_rows) else "diagnostic_only",
        )

    def can_use_for_outer_ik(self) -> bool:
        """Return whether this candidate may be used as a default outer-IK sensitivity."""
        return False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe interface snapshot."""
        decision = self.decision()
        evidence = self.evidence_summary()
        return {
            "decision": decision.__dict__,
            "evidence": evidence.__dict__,
            "outer_ik_default_allowed": self.can_use_for_outer_ik(),
            "required_label": "受信任分支条件",
        }
