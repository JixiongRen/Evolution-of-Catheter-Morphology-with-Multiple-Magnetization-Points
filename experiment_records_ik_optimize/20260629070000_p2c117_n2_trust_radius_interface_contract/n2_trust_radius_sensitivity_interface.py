"""Trust-radius guarded interface for the n2 position sensitivity."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
P2C107_DIR = ROOT / "experiment_records_ik_optimize/20260629013000_p2c107_n2_guarded_interface_candidate"
if str(P2C107_DIR) not in sys.path:
    sys.path.insert(0, str(P2C107_DIR))

from n2_guarded_sensitivity_interface import N2GuardedSensitivityInterface, ReuseDecision  # noqa: E402


@dataclass(frozen=True)
class StepDecision:
    """Decision for one proposed local current step."""

    accepted: bool
    reasons: tuple[str, ...]
    current_step_norm_A: float


class N2TrustRadiusSensitivityInterface(N2GuardedSensitivityInterface):
    """Expose n2 position sensitivity with an explicit linear trust radius."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.radius_summary = self._read("experiment_records_ik_optimize/20260629062000_p2c116_n2_calibrated_trust_radius/summary.json")

    def max_linear_current_step_A(self) -> float:
        """Return the largest audited current-step radius for local linear use."""
        passed = [
            float(radius)
            for radius, row in self.radius_summary["by_radius"].items()
            if int(row["passed"]) == int(row["total"])
            and float(row["max_rel_err"]) <= self.contract["gate_limits"]["linearization_rel_err_max"]
            and float(row["max_centerline_change_m"]) <= 1.0e-3
        ]
        return max(passed) if passed else 0.0

    def physical_current_step_budget_A(self) -> float:
        """Return the engineering current budget that must not be used as a linear radius."""
        return 5.0

    def trust_radius_evidence(self) -> dict[str, Any]:
        """Return audited trust-radius evidence."""
        return {
            "source_experiment": self.radius_summary["experiment_id"],
            "status": self.radius_summary["status"],
            "tested_cases": self.radius_summary["tested_cases"],
            "pass_count": self.radius_summary["pass_count"],
            "by_radius": self.radius_summary["by_radius"],
            "max_linear_current_step_A": self.max_linear_current_step_A(),
            "physical_current_step_budget_A": self.physical_current_step_budget_A(),
            "interpretation": "线性信赖域半径不同于物理电流预算；外层 IK 单步线性外推必须使用前者。",
        }

    def validate_step_request(self, request: dict[str, Any]) -> StepDecision:
        """Validate a local current-step request against cache and trust-radius guards."""
        reasons: list[str] = []
        reuse: ReuseDecision = self.validate_request(request)
        reasons.extend(reuse.reasons)
        if request.get("uses_physical_budget_as_linear_radius", False):
            reasons.append("physical_budget_used_as_linear_radius")
        if request.get("step_type") != "current_increment":
            reasons.append("step_type_mismatch")
        dI = request.get("dI_A")
        try:
            dI_array = np.asarray(dI, dtype=float).reshape(-1)
            current_norm = float(np.linalg.norm(dI_array))
        except (TypeError, ValueError):
            dI_array = np.array([], dtype=float)
            current_norm = float("nan")
            reasons.append("dI_parse_failed")
        if dI_array.shape != (8,):
            reasons.append("dI_shape_mismatch")
        max_radius = self.max_linear_current_step_A()
        if not np.isfinite(current_norm) or current_norm > max_radius + 1.0e-15:
            reasons.append("linear_trust_radius_exceeded")
        return StepDecision(accepted=not reasons, reasons=tuple(reasons), current_step_norm_A=current_norm)

    def can_use_for_default_outer_ik(self) -> bool:
        """Return whether this derivative may be used as the default outer-IK model."""
        return False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe interface snapshot."""
        payload = super().as_dict()
        payload["trust_radius"] = self.trust_radius_evidence()
        payload["required_step_request"] = {
            **payload["required_request"],
            "step_type": "current_increment",
            "dI_A": "8-vector with norm <= max_linear_current_step_A",
            "uses_physical_budget_as_linear_radius": False,
        }
        payload["forbidden_scope"] += "、将 5 A 物理电流预算当作单步线性信赖域"
        return payload
