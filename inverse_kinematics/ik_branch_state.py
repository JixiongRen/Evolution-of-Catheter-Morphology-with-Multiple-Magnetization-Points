"""Branch-state persistence utilities for supervised IK diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jax
import numpy as np


def array_sha256(value: Any) -> Optional[str]:
    """Return a stable SHA-256 digest for an array-like value."""
    if value is None:
        return None
    arr = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def json_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-serializable metadata."""
    payload = json.dumps(_jsonify(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_git_commit(cwd: Path) -> str:
    """Return the current Git commit or ``unknown`` when unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return "unknown"


def _jsonify(value: Any) -> Any:
    """Convert common numerical objects to JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return np.asarray(jax.device_get(value)).tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return repr(value)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonify(data), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonify(item), ensure_ascii=False, sort_keys=True))
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _scales_to_dict(scales: Any) -> dict[str, float]:
    return {
        "L_ref": float(scales.L_ref),
        "F_ref": float(scales.F_ref),
        "M_ref": float(scales.M_ref),
    }


def _lm_stats_to_dict(lm_stats: Any) -> dict[str, Any]:
    if lm_stats is None:
        return {}
    linear_info = getattr(lm_stats, "linear_info", None)
    return {
        "ok_strict": bool(getattr(lm_stats, "ok_strict", False)),
        "normE": float(getattr(lm_stats, "normE", np.nan)),
        "cost": float(getattr(lm_stats, "cost", np.nan)),
        "lam": float(getattr(lm_stats, "lam", np.nan)),
        "jac_method": str(getattr(lm_stats, "jac_method", "")),
        "stop_reason": str(getattr(lm_stats, "stop_reason", "")),
        "n_iter": int(getattr(lm_stats, "n_iter", -1)),
        "linear_solver": str(getattr(lm_stats, "linear_solver", "unknown")),
        "linear_info": {} if linear_info is None else _jsonify(linear_info),
        "has_dense_jacobian": getattr(lm_stats, "J", None) is not None,
    }


@dataclass
class BranchStateRecord:
    """Manifest entry returned after one saved IK/FK evaluation."""

    eval_id: str
    npz_path: Path
    z_hash: str
    warm_start_hash: Optional[str]
    centerline_hash: str


class BranchStateRecorder:
    """Persist every outer IK candidate evaluation for branch diagnostics."""

    def __init__(
        self,
        *,
        run_dir: Path,
        enabled: bool,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "branch_states"
        self.states_dir = self.root / "states"
        self.manifest_path = self.root / "branch_state_manifest.jsonl"
        self.gate_path = self.root / "branch_gate.json"
        self._counter = 0
        if self.enabled:
            self.states_dir.mkdir(parents=True, exist_ok=True)
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            _atomic_write_json(self.root / "metadata.json", metadata or {})
            _atomic_write_json(
                self.gate_path,
                {
                    "protocol": "state_snapshot_created",
                    "terminal_reload_gate_mm": 0.1,
                    "centerline_reload_gate_mm": 0.5,
                    "roundtrip_terminal_gate_mm": 0.25,
                    "roundtrip_centerline_gate_mm": 0.5,
                    "same_branch_requires": [
                        "terminal_reload",
                        "stepwise_trajectory",
                        "centerline_geometry",
                        "control_roundtrip",
                    ],
                    "status": "snapshots_ready_for_replay",
                },
            )

    def save(
        self,
        *,
        phase: str,
        iter_idx: int,
        pack: Any,
        p_des: Any,
        centerline_m: Any,
        centerline_relpath: Optional[str],
        status: str,
        accept: bool,
        parent_eval_id: Optional[str],
        outer: Optional[dict[str, Any]] = None,
    ) -> Optional[BranchStateRecord]:
        """Save one FK evaluation state and append its manifest row."""
        if not self.enabled:
            return None
        eval_id = f"eval_{self._counter:06d}"
        self._counter += 1

        z_star = np.asarray(jax.device_get(pack.z_star_bar), dtype=np.float64)
        warm_start = None
        if getattr(pack, "warm_start_z_bar", None) is not None:
            warm_start = np.asarray(jax.device_get(pack.warm_start_z_bar), dtype=np.float64)
        centerline = np.asarray(jax.device_get(centerline_m), dtype=np.float64)
        p_true = np.asarray(jax.device_get(pack.p_tip_true), dtype=np.float64).reshape(3)
        p_flex = np.asarray(jax.device_get(pack.p_tip), dtype=np.float64).reshape(3)
        p_target = np.asarray(jax.device_get(p_des), dtype=np.float64).reshape(3)
        I_A = np.asarray(jax.device_get(pack.I), dtype=np.float64).reshape(-1)
        E_star = None
        if getattr(pack, "lm_stats", None) is not None and getattr(pack.lm_stats, "E_star", None) is not None:
            E_star = np.asarray(jax.device_get(pack.lm_stats.E_star), dtype=np.float64)

        z_hash = array_sha256(z_star)
        warm_hash = array_sha256(warm_start)
        centerline_hash = array_sha256(centerline)
        npz_rel = Path("states") / f"{eval_id}_{phase}_iter_{int(iter_idx):04d}.npz"
        npz_path = self.root / npz_rel
        payload = {
            "z_star_bar": z_star,
            "centerline_m": centerline,
            "terminal_true_m": p_true,
            "flex_tip_m": p_flex,
            "target_m": p_target,
            "I_A": I_A,
            "x_m": np.asarray([float(pack.x)], dtype=np.float64),
            "scales": np.asarray(
                [
                    float(pack.params.scales.L_ref),
                    float(pack.params.scales.F_ref),
                    float(pack.params.scales.M_ref),
                ],
                dtype=np.float64,
            ),
        }
        if warm_start is not None:
            payload["warm_start_z_bar"] = warm_start
        if E_star is not None:
            payload["E_star"] = E_star
        np.savez(npz_path, **payload)

        manifest = {
            "eval_id": eval_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": str(phase),
            "iter": int(iter_idx),
            "status": str(status),
            "accept": bool(accept),
            "parent_eval_id": parent_eval_id,
            "npz_path": str(npz_rel),
            "z_star_hash": z_hash,
            "warm_start_hash": warm_hash,
            "centerline_hash": centerline_hash,
            "centerline_npy": centerline_relpath,
            "I_A": I_A.tolist(),
            "x_m": float(pack.x),
            "terminal_true_m": p_true.tolist(),
            "flex_tip_m": p_flex.tolist(),
            "target_m": p_target.tolist(),
            "terminal_error_mm": float(np.linalg.norm(p_true - p_target) * 1000.0),
            "flex_tip_error_mm": float(np.linalg.norm(p_flex - p_target) * 1000.0),
            "classification": str(pack.cls),
            "ok_strict": bool(pack.ok_strict),
            "normE": float(pack.normE),
            "M_list": [int(v) for v in pack.params.M_list],
            "L1_dim_m": float(pack.params.L1_dim),
            "scales": _scales_to_dict(pack.params.scales),
            "lm_stats": _lm_stats_to_dict(pack.lm_stats),
            "outer": {} if outer is None else _jsonify(outer),
        }
        _append_jsonl(self.manifest_path, manifest)
        return BranchStateRecord(
            eval_id=eval_id,
            npz_path=npz_path,
            z_hash=str(z_hash),
            warm_start_hash=warm_hash,
            centerline_hash=str(centerline_hash),
        )
