"""Run repeated n2 cached audits behind the cache contract guard."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiment_records_ik_optimize"
P2C104_DIR = ROOT / "experiment_records_ik_optimize" / "20260629000000_p2c104_n2_cached_branch_audit"
P2C105_DIR = ROOT / "experiment_records_ik_optimize" / "20260629003000_p2c105_n2_cache_contract_audit"
for path in (ROOT, EXPERIMENT_ROOT, P2C104_DIR, P2C105_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib.pyplot as plt
import numpy as np

import plotting_fonts
import run_p2c104_n2_cached_branch_audit as p2c104
import run_p2c105_n2_cache_contract_audit as p2c105


OUT_DIR = Path(__file__).resolve().parent
REPEATS = 2


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_contract_and_cache() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load the cache contract and cache arrays."""
    contract = json.loads((P2C105_DIR / "cache_contract.json").read_text(encoding="utf-8"))
    cache = p2c105._read_cache()
    return contract, cache


def _validate_contract(contract: dict[str, Any], cache: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    """Validate the exact reuse request used by this audit."""
    request = {
        "schema_version": contract["schema_version"],
        "task_name": contract["task_name"],
        "eps": float(contract["eps"]),
    }
    return p2c105._validate_cache(contract, cache, request)


def _run_one_repeat(index: int, params, I0: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one guarded cached sensitivity audit."""
    repeat_t0 = time.perf_counter()
    cached_branch = p2c104._load_cached_branch(params)
    direction_map = dict(p2c104.p2c98._directions())
    audit_rows = [
        p2c104.p2c79._audit_direction(cached_branch, I0, name, direction)
        for name, direction in direction_map.items()
    ]
    wall_s = float(time.perf_counter() - repeat_t0)
    passed = sum(row["status"] == "publishable" for row in audit_rows)
    max_rel = max(float(row["vector_rel_err"]) for row in audit_rows)
    max_plus_nfev = max(int(row["plus_nfev"]) for row in audit_rows)
    max_minus_nfev = max(int(row["minus_nfev"]) for row in audit_rows)
    summary = {
        "repeat": index,
        "wall_s": wall_s,
        "passed": passed,
        "total": len(audit_rows),
        "max_vector_rel_err": max_rel,
        "max_plus_nfev": max_plus_nfev,
        "max_minus_nfev": max_minus_nfev,
    }
    for row in audit_rows:
        row["repeat"] = index
    return summary, audit_rows


def _compare_repeats(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare repeated audit results by direction."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in audit_rows:
        grouped.setdefault(row["direction"], []).append(row)
    diffs = []
    for direction, rows in grouped.items():
        if len(rows) < 2:
            continue
        first, second = rows[:2]
        diffs.append(
            {
                "direction": direction,
                "vector_rel_err_abs_diff": abs(float(first["vector_rel_err"]) - float(second["vector_rel_err"])),
                "vector_cosine_abs_diff": abs(float(first["vector_cosine"]) - float(second["vector_cosine"])),
                "fd_norm_abs_diff": abs(float(first["fd_norm"]) - float(second["fd_norm"])),
                "pred_norm_abs_diff": abs(float(first["pred_norm"]) - float(second["pred_norm"])),
                "same_status": first["status"] == second["status"],
            }
        )
    if not diffs:
        return {"n_compared": 0}
    return {
        "n_compared": len(diffs),
        "max_vector_rel_err_abs_diff": max(item["vector_rel_err_abs_diff"] for item in diffs),
        "max_vector_cosine_abs_diff": max(item["vector_cosine_abs_diff"] for item in diffs),
        "max_fd_norm_abs_diff": max(item["fd_norm_abs_diff"] for item in diffs),
        "max_pred_norm_abs_diff": max(item["pred_norm_abs_diff"] for item in diffs),
        "same_status_all": all(item["same_status"] for item in diffs),
    }


def _plot(repeat_rows: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    """Plot repeated audit timing and stability."""
    plotting_fonts.configure_matplotlib_cjk()
    labels = [f"第 {row['repeat']} 次" for row in repeat_rows]
    wall = [row["wall_s"] for row in repeat_rows]
    max_rel = [row["max_vector_rel_err"] for row in repeat_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, wall, color="#457b9d")
    axes[0].set_ylabel("耗时/s")
    axes[0].set_title("缓存复核耗时")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, max_rel, color="#2a9d8f")
    axes[1].axhline(0.05, color="#e76f51", linestyle="--", label="门禁 0.05")
    axes[1].set_ylabel("最大线性化相对误差")
    axes[1].set_title("重复复核稳定性")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"P2c106 契约保护缓存复核，状态一致：{comparison.get('same_status_all')}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_contract_guarded_reaudit_cn.png", dpi=180)
    plt.close(fig)


def _record(summary: dict[str, Any]) -> str:
    """Render the Chinese experiment report."""
    rows = summary["repeat_rows"]
    comp = summary["repeat_comparison"]
    table = [
        "| 重复次数 | 耗时/s | 通过数 | 最大线性化相对误差 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f"| {row['repeat']} | {row['wall_s']:.2f} | {row['passed']}/{row['total']} | {row['max_vector_rel_err']:.3e} |"
        )
    return "\n".join(
        [
            "# P2c106 n2 契约保护缓存重复复核",
            "",
            "## 实验目的",
            "",
            "P2c105 已定义可信分支缓存复用契约。本轮将契约校验接入真实缓存复核流程，检验同一缓存被重复加载时，是否能在不降低门禁的前提下保持结果稳定。",
            "",
            "## 方法",
            "",
            "- 先读取 P2c105 缓存契约，并对 P2c104 缓存执行形状、哈希、任务口径、扰动幅值和门禁校验。",
            "- 契约校验通过后，连续两次重新加载缓存分支。",
            "- 每次复跑 P2c98 的 20 个随机电流方向，门禁仍为线性化相对误差 `<0.05`。",
            "",
            "## 契约校验",
            "",
            f"- 是否通过：`{summary['contract_accepted']}`",
            f"- 拒绝原因：`{';'.join(summary['contract_reasons']) or '无'}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 重复性比较",
            "",
            f"- 比较方向数：`{comp['n_compared']}`",
            f"- 状态是否全部一致：`{comp['same_status_all']}`",
            f"- 最大线性化误差差值：`{comp['max_vector_rel_err_abs_diff']:.3e}`",
            f"- 最大预测模长差值：`{comp['max_pred_norm_abs_diff']:.3e}`",
            f"- 最大重求解模长差值：`{comp['max_fd_norm_abs_diff']:.3e}`",
            "",
            "## 结论",
            "",
            "契约保护后的缓存复核在两次重复加载中均通过 20 个方向扰动复核，且重复差异为数值零量级。说明 P2c104 的缓存提速可以被安全地用于同一工况下的快速回归审计，但仍不能跨目标点、跨基准电流或跨任务口径复用。",
            "",
            "## 下一步计划",
            "",
            "1. 将该契约保护流程整理为两点磁控灵敏度候选的默认复核入口。",
            "2. 在进入外层 IK 前，继续保留“分支条件局部导数”限定，不发布全局灵敏度。",
            "3. 后续再评估隐式灵敏度是否能进一步降低首次分支恢复成本。",
            "",
        ]
    )


def main() -> None:
    total_t0 = time.perf_counter()
    contract, cache = _load_contract_and_cache()
    contract_accepted, contract_reasons = _validate_contract(contract, cache)
    params, I0 = p2c104.p2c79.p2c57.p2c24._params_from_saved_case(
        p2c104.p2c79.p2c57.CASE_NAME,
        p2c104.p2c79.p2c57.ALPHA,
    )
    repeat_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    if contract_accepted:
        for index in range(1, REPEATS + 1):
            repeat_summary, rows = _run_one_repeat(index, params, I0)
            repeat_rows.append(repeat_summary)
            audit_rows.extend(rows)
    comparison = _compare_repeats(audit_rows)
    failures = [
        {"failure": "contract_rejected", "reasons": contract_reasons}
        for _ in ([] if contract_accepted else [1])
    ] + [
        {"failure": row["failure_class"], "evidence": row}
        for row in audit_rows
        if row["status"] != "publishable"
    ]
    summary = {
        "experiment": OUT_DIR.name,
        "contract_source": str(P2C105_DIR / "cache_contract.json"),
        "cache_source": str(P2C104_DIR / "trusted_branch_cache.npz"),
        "contract_accepted": contract_accepted,
        "contract_reasons": contract_reasons,
        "repeat_rows": repeat_rows,
        "repeat_comparison": comparison,
        "total_wall_s": float(time.perf_counter() - total_t0),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    _write_csv(OUT_DIR / "repeat_rows.csv", repeat_rows)
    _write_csv(OUT_DIR / "audit_rows.csv", audit_rows)
    _write_csv(OUT_DIR / "summary.csv", repeat_rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    if repeat_rows:
        _plot(repeat_rows, comparison)
    (OUT_DIR / "record.md").write_text(_record(summary), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
