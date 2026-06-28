"""Diagnostic outer-IK policy for the guarded n2 position sensitivity."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
P2C117_DIR = ROOT / "experiment_records_ik_optimize/20260629070000_p2c117_n2_trust_radius_interface_contract"
if str(P2C117_DIR) not in sys.path:
    sys.path.insert(0, str(P2C117_DIR))

from n2_trust_radius_sensitivity_interface import N2TrustRadiusSensitivityInterface  # noqa: E402


@dataclass(frozen=True)
class PolicyDecision:
    """Decision for one local task-space target offset."""

    action: str
    reason: str
    estimated_steps: int
    required_current_norm_A: float
    first_step_norm_A: float
    controllability_ratio: float
    weak_mode_fraction: float


class N2DiagnosticOuterIKPolicy:
    """Classify n2 local position targets before any outer-IK execution."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.interface = N2TrustRadiusSensitivityInterface(root)
        cache = np.load(root / "experiment_records_ik_optimize/20260629000000_p2c104_n2_cached_branch_audit/trusted_branch_cache.npz")
        self.S = np.asarray(cache["S"], dtype=float)
        self.U, self.singular_values, self.Vt = np.linalg.svd(self.S, full_matrices=False)
        self.radius_A = self.interface.max_linear_current_step_A()

    def classify(self, target_offset_m: np.ndarray, max_steps: int = 8) -> PolicyDecision:
        """Classify a local target offset using the audited trust radius."""
        target = np.asarray(target_offset_m, dtype=float).reshape(3)
        dI = np.linalg.pinv(self.S, rcond=1.0e-12) @ target
        required = float(np.linalg.norm(dI))
        estimated_steps = int(np.ceil(required / max(self.radius_A, 1.0e-18)))
        coeff = self.U.T @ target
        target_norm = float(np.linalg.norm(target))
        weak_mode_fraction = float(abs(coeff[-1]) / max(np.linalg.norm(coeff), 1.0e-18))
        strongest_reachable = float(self.singular_values[0] * self.radius_A * max_steps)
        controllability_ratio = float(target_norm / max(strongest_reachable, 1.0e-18))
        first_step_norm = min(required, self.radius_A)

        if estimated_steps <= 4 and weak_mode_fraction < 0.25:
            action = "快速局部校正"
            reason = "目标主要位于强可控方向，预计少量小步可明显降低误差"
        elif estimated_steps <= max_steps and weak_mode_fraction < 0.60:
            action = "慢速诊断推进"
            reason = "目标可在信赖域内推进，但方向混合或步数较多，应逐步复查"
        else:
            action = "拒绝直接外层 IK"
            reason = "目标含弱可控成分过高或预计步数超过诊断上限，应重新规划任务或更换工作点"

        return PolicyDecision(
            action=action,
            reason=reason,
            estimated_steps=estimated_steps,
            required_current_norm_A=required,
            first_step_norm_A=first_step_norm,
            controllability_ratio=controllability_ratio,
            weak_mode_fraction=weak_mode_fraction,
        )

    def first_step(self, target_offset_m: np.ndarray) -> np.ndarray:
        """Return the first trust-radius-limited current step."""
        target = np.asarray(target_offset_m, dtype=float).reshape(3)
        dI = np.linalg.pinv(self.S, rcond=1.0e-12) @ target
        norm = float(np.linalg.norm(dI))
        if norm > self.radius_A:
            dI = dI * (self.radius_A / norm)
        return np.asarray(dI, dtype=float)

    def as_dict(self) -> dict:
        """Return a JSON-safe policy snapshot."""
        return {
            "task": "n2 三维末端位置",
            "mode": "诊断模式外层 IK 策略",
            "max_linear_current_step_A": self.radius_A,
            "physical_current_budget_A": self.interface.physical_current_step_budget_A(),
            "singular_values": self.singular_values.tolist(),
            "allowed": "可信分支、小电流半径内、位置任务的小步诊断推进",
            "forbidden": "完整位姿、跨分支、跨基准电流、大步线性外推、默认生产 IK",
        }
