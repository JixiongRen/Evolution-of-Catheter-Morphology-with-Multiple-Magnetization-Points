"""Directional engineering policy for trusted n2 position sensitivity."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
P2C119_DIR = ROOT / "experiment_records_ik_optimize/20260629080000_p2c119_n2_diagnostic_outer_ik_policy"
if str(P2C119_DIR) not in sys.path:
    sys.path.insert(0, str(P2C119_DIR))

from n2_diagnostic_outer_ik_policy import N2DiagnosticOuterIKPolicy  # noqa: E402


@dataclass(frozen=True)
class DirectionalPolicyDecision:
    """Decision for one local n2 target request."""

    action: str
    reason: str
    target_norm_nm: float
    estimated_steps: int
    weak_mode_fraction: float
    required_current_norm_A: float
    allowed_for_default_outer_ik: bool


class N2DirectionalEngineeringPolicy:
    """Expose conservative engineering rules from P2c122-P2c124."""

    strong_closed_nm = 50.0
    weak_diagnostic_nm = 1.0
    weak_fraction_limit = 0.25
    mixed_fraction_limit = 0.60
    max_diagnostic_steps = 12

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.base_policy = N2DiagnosticOuterIKPolicy(root)
        self.S = self.base_policy.S
        self.U = self.base_policy.U
        self.radius_A = self.base_policy.radius_A
        self.p2c122 = self._read("experiment_records_ik_optimize/20260629100000_p2c122_n2_workspace_direction_scan/summary.json")
        self.p2c123 = self._read("experiment_records_ik_optimize/20260629110000_p2c123_n2_direction_reopt_closure/summary.json")
        self.p2c124 = self._read("experiment_records_ik_optimize/20260629120000_p2c124_n2_multistep_direction_closure/summary.json")

    def _read(self, relative_path: str) -> dict[str, Any]:
        import json

        return json.loads((self.root / relative_path).read_text(encoding="utf-8"))

    def classify(self, target_offset_m: np.ndarray) -> DirectionalPolicyDecision:
        """Classify a local target offset without executing IK."""
        target = np.asarray(target_offset_m, dtype=float).reshape(3)
        target_norm = float(np.linalg.norm(target))
        target_norm_nm = target_norm * 1.0e9
        dI = np.linalg.pinv(self.S, rcond=1.0e-12) @ target
        required = float(np.linalg.norm(dI))
        estimated_steps = int(np.ceil(required / max(self.radius_A, 1.0e-18)))
        coeff = self.U.T @ target
        coeff_norm = max(float(np.linalg.norm(coeff)), 1.0e-18)
        weak_fraction = float(abs(coeff[-1]) / coeff_norm)

        if target_norm <= 0.0:
            action = "无动作"
            reason = "目标位移为零"
        elif weak_fraction < self.weak_fraction_limit and target_norm_nm <= self.strong_closed_nm and estimated_steps <= 4:
            action = "强方向小目标闭环候选"
            reason = "P2c124 已验证强方向 50 nm 可在少量小步内闭合；每步仍需真实重求解复核"
        elif weak_fraction >= self.mixed_fraction_limit and target_norm_nm <= self.weak_diagnostic_nm:
            action = "弱方向诊断观察"
            reason = "弱方向单步导数可信，但 P2c124 显示推进效率极低；只允许诊断，不作为修正器"
        elif weak_fraction < self.mixed_fraction_limit and estimated_steps <= self.max_diagnostic_steps:
            action = "证据外谨慎诊断"
            reason = "目标未落入已验证闭环候选范围；只允许小步诊断，不得声明可闭环"
        else:
            action = "拒绝直接外层 IK"
            reason = "目标尺度、预计步数或弱方向成分超出已审计工程边界"

        return DirectionalPolicyDecision(
            action=action,
            reason=reason,
            target_norm_nm=target_norm_nm,
            estimated_steps=estimated_steps,
            weak_mode_fraction=weak_fraction,
            required_current_norm_A=required,
            allowed_for_default_outer_ik=False,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy snapshot."""
        return {
            "task": "两点磁控可信分支下三维末端位置局部灵敏度",
            "allowed_scope": "同一工作点、同一可信分支、小电流信赖域内的位置任务诊断",
            "forbidden_scope": "完整末端位姿、跨分支、跨电流基点、默认生产外层 IK、弱方向快速推进",
            "default_outer_ik_allowed": False,
            "strong_direction_rule": f"强方向目标不超过 {self.strong_closed_nm:g} nm 可作为闭环候选，且每步真实重求解",
            "weak_direction_rule": f"弱方向不超过 {self.weak_diagnostic_nm:g} nm 仅诊断观察；更大弱方向目标拒绝直接推进",
            "max_linear_current_step_A": self.radius_A,
            "evidence": {
                "direction_scan": self.p2c122["experiment_id"],
                "single_step_closure": self.p2c123["experiment_id"],
                "multistep_closure": self.p2c124["experiment_id"],
                "strong_50nm_status": next(row for row in self.p2c124["target_summaries"] if row["target"] == "强方向_50nm")["status"],
                "weak_5nm_reduction_ratio": next(row for row in self.p2c124["target_summaries"] if row["target"] == "弱方向_5nm")["reduction_ratio"],
            },
        }
