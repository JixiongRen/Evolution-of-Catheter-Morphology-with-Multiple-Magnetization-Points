"""Audit n2 sensitivity using a cached trusted branch."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C79_DIR = ROOT / "experiment_records_ik_optimize" / "20260628020500_p2c79_branch_random_validation"
P2C98_DIR = ROOT / "experiment_records_ik_optimize" / "20260628213500_p2c98_n2_eps0125_random_current_batch"
for path in (ROOT, P2C79_DIR, P2C98_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib.pyplot as plt
import numpy as np

import run_p2c79_branch_random_validation as p2c79
import run_p2c98_n2_eps0125_random_current_batch as p2c98


OUT_DIR = Path(__file__).resolve().parent
EPS = 0.0125
CACHE_PATH = OUT_DIR / "trusted_branch_cache.npz"
P2C98_SUMMARY = P2C98_DIR / "summary.json"


def _save_cache(branch: dict, I0: np.ndarray, target_p: np.ndarray) -> None:
    """Save the minimum trusted-branch state required for direction audits."""
    np.savez_compressed(
        CACHE_PATH,
        I0=np.asarray(I0, dtype=float),
        target_p=np.asarray(target_p, dtype=float),
        y=np.asarray(branch["y"], dtype=float),
        y_ref=np.asarray(branch["y_ref"], dtype=float),
        basis=np.asarray(branch["basis"], dtype=float),
        S=np.asarray(branch["S"], dtype=float),
        singular_values=np.asarray(branch["singular_values"], dtype=float),
    )


def _load_cached_branch(params) -> dict:
    """Load a trusted branch and rebuild lightweight functions."""
    cached = np.load(CACHE_PATH)
    target_p = np.asarray(cached["target_p"], dtype=float)
    y_ref = np.asarray(cached["y_ref"], dtype=float)
    _args, funcs = p2c79.p2c69._make_funcs(params, target_p, y_ref, p2c79.p2c69.STAGES[-1][0])
    return {
        "label": "cached_trusted_seed",
        "y": np.asarray(cached["y"], dtype=float),
        "y_ref": y_ref,
        "basis": np.asarray(cached["basis"], dtype=float),
        "S": np.asarray(cached["S"], dtype=float),
        "funcs": funcs,
    }


def _plot(timing: dict) -> None:
    """Plot full audit versus cached audit timing."""
    labels = ["P2c98 全量", "P2c104 含建缓存", "P2c104 仅缓存复核"]
    values = [timing["p2c98_wall_s"], timing["total_wall_s"], timing["cached_audit_wall_s"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, values, color=["#457b9d", "#e76f51", "#2a9d8f"])
    ax.set_ylabel("耗时/s")
    ax.set_title("P2c104 n2 可信分支缓存复核耗时")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_cached_branch_audit_timing_cn.png", dpi=180)
    plt.close(fig)


def _record(timing: dict, audit_rows: list[dict], branch_row: dict) -> str:
    """Render the Chinese experiment report."""
    passed = sum(row["status"] == "publishable" for row in audit_rows)
    max_err = max(row["vector_rel_err"] for row in audit_rows)
    cache_speedup = timing["p2c98_wall_s"] / timing["cached_audit_wall_s"]
    table = [
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| P2c98 全量耗时/s | {timing['p2c98_wall_s']:.2f} |",
        f"| P2c104 含建缓存总耗时/s | {timing['total_wall_s']:.2f} |",
        f"| 可信分支恢复耗时/s | {timing['recover_branch_wall_s']:.2f} |",
        f"| 仅缓存复核耗时/s | {timing['cached_audit_wall_s']:.2f} |",
        f"| 仅缓存复核相对 P2c98 加速比 | {cache_speedup:.2f} |",
        f"| 随机方向通过数 | {passed}/{len(audit_rows)} |",
        f"| 最大线性化相对误差 | {max_err:.3e} |",
    ]
    return "\n".join(
        [
            "# P2c104 n2 可信分支缓存复核记录",
            "",
            "## 实验目的",
            "",
            "P2c103 证明 n2 灵敏度审计的主要耗时来自受信任分支恢复，而不是方向扰动重求解。本轮构建一次可信分支缓存，并从缓存重新载入分支状态、有效子空间和任务灵敏度矩阵，复跑 P2c98 的 20 个随机电流方向，检验缓存复核能否在不降低门禁的前提下提速。",
            "",
            "## 方法",
            "",
            "- 先按原流程恢复受信任分支，并保存 `y`、`y_ref`、`basis`、`S`、`I0`、`target_p`。",
            "- 随后重新载入缓存，重建轻量残差函数，跳过受信任分支恢复。",
            "- 使用 P2c98 相同随机方向、`eps=0.0125` 和线性化误差 `<0.05` 门禁。",
            "",
            "## 基线门禁",
            "",
            f"- 一阶最优性：`{branch_row['stationarity_norm']:.3e}`",
            f"- 载荷传播比例：`{branch_row['force_ratio']:.3e}`",
            f"- 任务条件数：`{branch_row['task_condition']:.3e}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"缓存复核中 `{passed}` / `{len(audit_rows)}` 个方向通过发布级门禁，最大线性化相对误差为 `{max_err:.3e}`。仅计算缓存复核阶段时，耗时从 P2c98 的 `{timing['p2c98_wall_s']:.2f} s` 降至 `{timing['cached_audit_wall_s']:.2f} s`，说明缓存受信任分支可以显著降低常规复核成本。",
            "",
            "需要注意，首次建立缓存仍需完整分支恢复，因此该方法适用于同一工况的重复审计、代码修改后的回归验证和外层策略迭代；不能把缓存建立成本从一次性发布审计中删除。",
            "",
            "该实验不改变 n2 的分支条件限定，也不允许默认接入外层 IK。",
            "",
            "## 下一步计划",
            "",
            "1. 将缓存接口整理为可复用工程入口，明确缓存失效条件。",
            "2. 若模型、目标点、磁化配置、电流基点或门禁改变，必须重建缓存。",
            "3. 后续效率优化重点转为缓存生命周期管理和批量方向复核。",
            "",
        ]
    )


def main() -> None:
    p2c79.OUT_DIR = OUT_DIR
    p2c79.EPS = EPS
    total_t0 = time.perf_counter()
    p2c98_summary = json.loads(P2C98_SUMMARY.read_text(encoding="utf-8"))
    params, I0 = p2c79.p2c57.p2c24._params_from_saved_case(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA)
    target_p, y_slack = p2c79.p2c57.p2c24._load_saved_target_and_seed(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA)
    y_seed = p2c79.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c79.p2c57.Args.steps_per_flex)
    recover_t0 = time.perf_counter()
    branch = p2c79._recover_branch(params, I0, target_p, y_seed, "trusted_seed")
    recover_wall_s = float(time.perf_counter() - recover_t0)
    branch_row = p2c79._branch_row(branch, branch)
    _save_cache(branch, I0, target_p)

    cached_t0 = time.perf_counter()
    cached_branch = _load_cached_branch(params)
    direction_map = dict(p2c98._directions())
    audit_rows = [p2c79._audit_direction(cached_branch, I0, name, direction) for name, direction in direction_map.items()]
    cached_audit_wall_s = float(time.perf_counter() - cached_t0)
    total_wall_s = float(time.perf_counter() - total_t0)
    timing = {
        "p2c98_wall_s": float(p2c98_summary["wall_s"]),
        "recover_branch_wall_s": recover_wall_s,
        "cached_audit_wall_s": cached_audit_wall_s,
        "total_wall_s": total_wall_s,
        "cache_path": str(CACHE_PATH),
    }
    p2c79._write_csv(OUT_DIR / "branch_rows.csv", [branch_row])
    p2c79._write_csv(OUT_DIR / "audit_rows.csv", audit_rows)
    p2c79._write_csv(OUT_DIR / "summary.csv", [branch_row] + audit_rows)
    failures = [
        {"failure": row["failure_class"], "evidence": row}
        for row in audit_rows
        if row["status"] != "publishable"
    ]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(p2c79._jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "eps": EPS,
        "timing": timing,
        "branch_row": branch_row,
        "audit_rows": audit_rows,
        "cache_arrays": {key: list(value.shape) for key, value in np.load(CACHE_PATH).items()},
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(p2c79._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(timing)
    (OUT_DIR / "record.md").write_text(_record(timing, audit_rows, branch_row), encoding="utf-8")
    print(json.dumps(p2c79._jsonable({"timing": timing, "failures": failures}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
