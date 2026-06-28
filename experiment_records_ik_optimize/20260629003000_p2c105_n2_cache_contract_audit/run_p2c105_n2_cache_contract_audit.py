"""Validate reuse rules for the n2 trusted-branch cache."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C104_DIR = ROOT / "experiment_records_ik_optimize" / "20260629000000_p2c104_n2_cached_branch_audit"
CACHE_PATH = P2C104_DIR / "trusted_branch_cache.npz"
P2C104_SUMMARY = P2C104_DIR / "summary.json"

SCHEMA_VERSION = "n2_trusted_branch_cache_v1"
SOURCE_EXPERIMENT = "20260629000000_p2c104_n2_cached_branch_audit"
TASK_NAME = "n2_position_only_branch_conditioned"
MAX_LINEARIZATION_REL_ERR = 0.05
MAX_STATIONARITY_NORM = 1.0e-5
MAX_FORCE_RATIO = 1.5
MAX_TASK_CONDITION = 1.0e7


def _jsonable(value: Any) -> Any:
    """Convert numpy values to JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _array_hash(array: np.ndarray) -> str:
    """Return a stable hash for one numeric array."""
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _read_cache() -> dict[str, np.ndarray]:
    """Read the local trusted-branch cache."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"cache missing: {CACHE_PATH}")
    with np.load(CACHE_PATH) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _build_contract(cache: dict[str, np.ndarray], summary: dict[str, Any]) -> dict[str, Any]:
    """Build the metadata contract required before cache reuse."""
    branch = summary["branch_row"]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_experiment": SOURCE_EXPERIMENT,
        "task_name": TASK_NAME,
        "eps": float(summary["eps"]),
        "cache_arrays": {key: {"shape": list(value.shape), "sha256": _array_hash(value)} for key, value in cache.items()},
        "gate_limits": {
            "stationarity_norm_max": MAX_STATIONARITY_NORM,
            "force_ratio_max": MAX_FORCE_RATIO,
            "task_condition_max": MAX_TASK_CONDITION,
            "linearization_rel_err_max": MAX_LINEARIZATION_REL_ERR,
        },
        "gate_values": {
            "stationarity_norm": float(branch["stationarity_norm"]),
            "force_ratio": float(branch["force_ratio"]),
            "task_condition": float(branch["task_condition"]),
            "random_direction_max_rel_err": max(float(row["vector_rel_err"]) for row in summary["audit_rows"]),
            "random_direction_passed": sum(row["status"] == "publishable" for row in summary["audit_rows"]),
            "random_direction_total": len(summary["audit_rows"]),
        },
        "reuse_scope": {
            "allowed": "same model, same task definition, same target point, same base current, same gate policy",
            "forbidden": "different model, target point, base current, magnet configuration, task dimension, gate limits, or branch label",
        },
    }


def _validate_cache(contract: dict[str, Any], cache: dict[str, np.ndarray], request: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate one cache reuse request against the contract."""
    reasons: list[str] = []
    if request.get("schema_version") != contract["schema_version"]:
        reasons.append("schema_version_mismatch")
    if request.get("task_name") != contract["task_name"]:
        reasons.append("task_name_mismatch")
    if abs(float(request.get("eps", np.nan)) - float(contract["eps"])) > 1e-15:
        reasons.append("eps_mismatch")

    arrays = contract["cache_arrays"]
    for key, spec in arrays.items():
        if key not in cache:
            reasons.append(f"missing_array:{key}")
            continue
        value = cache[key]
        if list(value.shape) != spec["shape"]:
            reasons.append(f"shape_mismatch:{key}")
        if _array_hash(value) != spec["sha256"]:
            reasons.append(f"hash_mismatch:{key}")

    gates = contract["gate_values"]
    limits = contract["gate_limits"]
    if gates["stationarity_norm"] > limits["stationarity_norm_max"]:
        reasons.append("stationarity_gate_failed")
    if gates["force_ratio"] > limits["force_ratio_max"]:
        reasons.append("force_ratio_gate_failed")
    if gates["task_condition"] > limits["task_condition_max"]:
        reasons.append("task_condition_gate_failed")
    if gates["random_direction_max_rel_err"] > limits["linearization_rel_err_max"]:
        reasons.append("linearization_gate_failed")
    if gates["random_direction_passed"] != gates["random_direction_total"]:
        reasons.append("direction_gate_failed")
    return not reasons, reasons


def _mutated_cache(cache: dict[str, np.ndarray], key: str, delta: float) -> dict[str, np.ndarray]:
    """Return a copied cache with one array perturbed."""
    copied = {name: value.copy() for name, value in cache.items()}
    copied[key] = copied[key].copy()
    copied[key].flat[0] += delta
    return copied


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot validation outcomes."""
    labels = [row["case"] for row in rows]
    values = [1 if row["accepted"] else 0 for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#2a9d8f" if value else "#e76f51" for value in values]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("是否允许复用")
    ax.set_title("P2c105 n2 可信分支缓存复用契约审计")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_cache_contract_validation_cn.png", dpi=180)
    plt.close(fig)


def _record(summary: dict[str, Any]) -> str:
    """Render the Chinese experiment report."""
    rows = summary["validation_rows"]
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    return "\n".join(
        [
            "# P2c105 n2 可信分支缓存复用契约审计",
            "",
            "## 实验目的",
            "",
            "P2c104 证明在同一两点磁控工作点下，复用可信分支缓存可以显著降低重复灵敏度复核耗时。本轮补充缓存复用契约，防止在目标点、电流基点、任务口径或门禁改变后误用旧缓存。",
            "",
            "## 方法",
            "",
            "- 从 P2c104 本地缓存读取 `I0`、`target_p`、`y`、`y_ref`、`basis`、`S` 与奇异值。",
            "- 为每个数组记录形状和哈希值，并记录任务口径、扰动幅值、门禁阈值和门禁实测值。",
            "- 构造一个正例和多组故意篡改负例，验证契约能否接受正确缓存并拒绝错误缓存。",
            "",
            "## 结果",
            "",
            f"- 正例通过数：`{len(accepted)}`",
            f"- 负例拒绝数：`{len(rejected)}`",
            f"- 审计耗时：`{summary['wall_s']:.3f} s`",
            f"- 基线最大线性化相对误差：`{summary['contract']['gate_values']['random_direction_max_rel_err']:.3e}`",
            f"- 基线载荷传播比例：`{summary['contract']['gate_values']['force_ratio']:.3e}`",
            "",
            "## 复用边界",
            "",
            "允许复用的前提是模型、任务定义、目标点、基准电流和门禁策略完全一致。若磁体配置、目标点、电流基点、任务维度、残差定义或门禁阈值发生变化，必须重建可信分支缓存。",
            "",
            "## 结论",
            "",
            "P2c105 给 P2c104 的缓存提速结果增加了工程约束。该缓存可以用于同一工况的快速回归复核，但不能跨工况、跨目标点或跨任务口径复用。该结论不扩大 n2 灵敏度的发布范围，仍限定为受信任分支条件下的三维末端位置局部灵敏度。",
            "",
            "## 下一步计划",
            "",
            "1. 将缓存契约接入后续复核脚本，默认先校验再加载。",
            "2. 对 n2 发布候选补充两次缓存复核重复性实验，确认缓存加载结果稳定。",
            "3. 再考虑把该接口整理进工程模块，而不是只停留在实验脚本。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    cache = _read_cache()
    p2c104_summary = json.loads(P2C104_SUMMARY.read_text(encoding="utf-8"))
    contract = _build_contract(cache, p2c104_summary)
    base_request = {
        "schema_version": SCHEMA_VERSION,
        "task_name": TASK_NAME,
        "eps": float(p2c104_summary["eps"]),
    }
    cases = [
        ("exact_cache", cache, base_request),
        ("changed_eps", cache, {**base_request, "eps": 0.01}),
        ("changed_schema", cache, {**base_request, "schema_version": "bad_schema"}),
        ("changed_task", cache, {**base_request, "task_name": "full_pose"}),
        ("changed_target_p", _mutated_cache(cache, "target_p", 1.0e-7), base_request),
        ("changed_I0", _mutated_cache(cache, "I0", 1.0e-6), base_request),
        ("changed_basis", _mutated_cache(cache, "basis", 1.0e-9), base_request),
        ("missing_S", {key: value for key, value in cache.items() if key != "S"}, base_request),
    ]
    rows: list[dict[str, Any]] = []
    for name, candidate_cache, request in cases:
        accepted, reasons = _validate_cache(contract, candidate_cache, request)
        rows.append({"case": name, "accepted": accepted, "reasons": ";".join(reasons) or "none"})
    wall_s = float(time.perf_counter() - t0)
    summary = {
        "experiment": OUT_DIR.name,
        "source_experiment": SOURCE_EXPERIMENT,
        "wall_s": wall_s,
        "contract": contract,
        "validation_rows": rows,
    }
    (OUT_DIR / "cache_contract.json").write_text(json.dumps(_jsonable(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "validation_rows.json").write_text(json.dumps(_jsonable(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "validation_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(row), ensure_ascii=False) + "\n" for row in rows if row["accepted"] != (row["case"] == "exact_cache")),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(summary), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "validation_rows": rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
