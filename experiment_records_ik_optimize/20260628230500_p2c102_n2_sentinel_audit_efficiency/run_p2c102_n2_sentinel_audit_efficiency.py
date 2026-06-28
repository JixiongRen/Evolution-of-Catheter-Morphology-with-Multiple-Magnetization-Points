"""Evaluate a sentinel-direction audit for n2 sensitivity efficiency."""

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
FULL_SUMMARY = P2C98_DIR / "summary.json"


def _select_sentinels(rows: list[dict]) -> list[str]:
    """Select representative risky directions from a full audit."""
    selectors = [
        max(rows, key=lambda row: row["vector_rel_err"])["direction"],
        min(rows, key=lambda row: row["vector_cosine"])["direction"],
        min(rows, key=lambda row: row["fd_norm"])["direction"],
        max(rows, key=lambda row: row["fd_norm"])["direction"],
        max(rows, key=lambda row: row["pred_norm"])["direction"],
    ]
    ordered: list[str] = []
    for name in selectors:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _plot(full_rows: list[dict], sentinel_rows: list[dict]) -> None:
    """Plot full versus sentinel audit metrics."""
    full_max = max(row["vector_rel_err"] for row in full_rows)
    sent_max = max(row["vector_rel_err"] for row in sentinel_rows)
    full_count = len(full_rows)
    sent_count = len(sentinel_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["全量审计", "哨兵审计"], [full_max, sent_max], color=["#457b9d", "#2a9d8f"])
    axes[0].axhline(p2c79.TASK_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("最大线性化误差")
    axes[1].bar(["全量审计", "哨兵审计"], [full_count, sent_count], color=["#457b9d", "#2a9d8f"])
    axes[1].set_ylabel("方向数")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c102 n2 哨兵方向审计效率评估")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_sentinel_audit_efficiency_cn.png", dpi=180)
    plt.close(fig)


def _record(full_data: dict, sentinel_names: list[str], sentinel_rows: list[dict], wall_s: float) -> str:
    """Render the Chinese experiment report."""
    full_rows = full_data["audit_rows"]
    full_wall = float(full_data["wall_s"])
    full_max = max(row["vector_rel_err"] for row in full_rows)
    sent_max = max(row["vector_rel_err"] for row in sentinel_rows)
    passed = sum(row["status"] == "publishable" for row in sentinel_rows)
    table = [
        "| 方向 | 状态 | 相对误差 | 方向余弦 | 有限差分范数 | 选择理由 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    reasons = {
        max(full_rows, key=lambda row: row["vector_rel_err"])["direction"]: "全量审计最大线性化误差",
        min(full_rows, key=lambda row: row["vector_cosine"])["direction"]: "全量审计最小方向余弦",
        min(full_rows, key=lambda row: row["fd_norm"])["direction"]: "全量审计最小有限差分响应",
        max(full_rows, key=lambda row: row["fd_norm"])["direction"]: "全量审计最大有限差分响应",
        max(full_rows, key=lambda row: row["pred_norm"])["direction"]: "全量审计最大预测响应",
    }
    for row in sentinel_rows:
        table.append(
            f"| {row['direction']} | {row['status']} | {row['vector_rel_err']:.3e} | "
            f"{row['vector_cosine']:.3e} | {row['fd_norm']:.3e} | {reasons.get(row['direction'], '去重保留')} |"
        )
    speed_ratio = full_wall / wall_s if wall_s > 0 else float("inf")
    return "\n".join(
        [
            "# P2c102 n2 哨兵方向审计效率评估记录",
            "",
            "## 实验目的",
            "",
            "P2c98 已完成 20 个随机电流方向的全量批量复核，但耗时约 530 秒。本轮从 P2c98 中选择少量高风险代表方向并重新求解，评估是否可作为常规回归审计的低成本哨兵集合。该实验不替代最终发布前的全量审计。",
            "",
            "## 哨兵选择规则",
            "",
            "从 P2c98 全量方向中选择：最大线性化误差、最小方向余弦、最小有限差分响应、最大有限差分响应、最大预测响应。若规则选中同一方向，则去重。",
            "",
            f"- 选中方向：`{', '.join(sentinel_names)}`",
            f"- 全量方向数：`{len(full_rows)}`",
            f"- 哨兵方向数：`{len(sentinel_rows)}`",
            "",
            "## 复核结果",
            "",
            *table,
            "",
            "## 效率比较",
            "",
            f"- P2c98 全量耗时：`{full_wall:.2f} s`",
            f"- P2c102 哨兵耗时：`{wall_s:.2f} s`",
            f"- 方向数降低：`{len(full_rows)}` -> `{len(sentinel_rows)}`",
            f"- 表观加速比：`{speed_ratio:.2f}`",
            f"- 全量最大误差：`{full_max:.3e}`",
            f"- 哨兵最大误差：`{sent_max:.3e}`",
            "",
            "## 结论",
            "",
            f"哨兵方向共有 `{passed}` / `{len(sentinel_rows)}` 个通过发布级门禁。但本轮未获得墙钟加速：方向数从 `{len(full_rows)}` 降到 `{len(sentinel_rows)}`，耗时仍与全量审计接近。这说明当前耗时主导项不是方向数量，而是受信任分支恢复、编译或固定系统构建成本。",
            "",
            "因此，哨兵方向集合只能作为风险代表方向集合保留，不能称为有效效率优化。真正的下一步应是缓存受信任分支、复用已构建的线性化系统，或把多个方向放入同一次分支恢复后的批量流程中。该结果不改变 n2 当前的分支条件限定，也不允许默认接入外层 IK。",
            "",
            "## 下一步计划",
            "",
            "1. 将哨兵集合作为后续代码修改后的风险代表方向，而不是独立快速门禁。",
            "2. 若未来哨兵失败，立即回到全量随机方向审计和方向性边界夹逼。",
            "3. 优先探索可信分支缓存和批量方向复用，降低分支恢复和固定系统构建成本。",
            "",
        ]
    )


def main() -> None:
    p2c79.OUT_DIR = OUT_DIR
    p2c79.EPS = EPS
    p2c98.OUT_DIR = OUT_DIR
    full_data = json.loads(FULL_SUMMARY.read_text(encoding="utf-8"))
    sentinel_names = _select_sentinels(full_data["audit_rows"])
    direction_map = dict(p2c98._directions())
    t0 = time.perf_counter()
    params, I0 = p2c79.p2c57.p2c24._params_from_saved_case(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA)
    target_p, y_slack = p2c79.p2c57.p2c24._load_saved_target_and_seed(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA)
    y_seed = p2c79.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c79.p2c57.Args.steps_per_flex)
    branch = p2c79._recover_branch(params, I0, target_p, y_seed, "trusted_seed")
    branch_row = p2c79._branch_row(branch, branch)
    sentinel_rows = [p2c79._audit_direction(branch, I0, name, direction_map[name]) for name in sentinel_names]
    wall_s = float(time.perf_counter() - t0)
    p2c79._write_csv(OUT_DIR / "branch_rows.csv", [branch_row])
    p2c79._write_csv(OUT_DIR / "sentinel_direction_rows.csv", sentinel_rows)
    p2c79._write_csv(OUT_DIR / "summary.csv", [branch_row] + sentinel_rows)
    failures = [
        {"failure": row["failure_class"], "evidence": row}
        for row in sentinel_rows
        if row["status"] != "publishable"
    ]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(p2c79._jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "eps": EPS,
        "sentinel_names": sentinel_names,
        "wall_s": wall_s,
        "full_wall_s": full_data["wall_s"],
        "speed_ratio": float(full_data["wall_s"] / wall_s),
        "branch_row": branch_row,
        "sentinel_rows": sentinel_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(p2c79._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(full_data["audit_rows"], sentinel_rows)
    (OUT_DIR / "record.md").write_text(_record(full_data, sentinel_names, sentinel_rows, wall_s), encoding="utf-8")
    print(json.dumps(p2c79._jsonable({"wall_s": wall_s, "failures": failures, "sentinel_names": sentinel_names}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
