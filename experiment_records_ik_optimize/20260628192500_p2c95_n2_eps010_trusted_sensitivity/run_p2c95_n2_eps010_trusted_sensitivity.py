"""Audit n2 trusted-branch task sensitivity at eps=0.010."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C79_DIR = ROOT / "experiment_records_ik_optimize" / "20260628020500_p2c79_branch_random_validation"
for path in (ROOT, P2C79_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib.pyplot as plt
import numpy as np

import run_p2c79_branch_random_validation as p2c79


OUT_DIR = Path(__file__).resolve().parent
EPS = 0.010


def _plot(audit_rows: list[dict]) -> None:
    """Plot eps=0.010 trusted-branch sensitivity diagnostics."""
    labels = [row["direction"] for row in audit_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].bar(x, [row["vector_rel_err"] for row in audit_rows], color="#457b9d")
    axes[0].axhline(p2c79.TASK_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("线性化相对误差")
    axes[1].bar(x, [max(0.0, row["vector_cosine"]) for row in audit_rows], color="#2a9d8f")
    axes[1].set_ylabel("方向余弦")
    axes[2].bar(x, [row["fd_norm"] for row in audit_rows], color="#e76f51")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("有限差分范数")
    axes[2].set_xticks(x, labels, rotation=30, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c95 n2 受信任分支 eps=0.010 灵敏度复核")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_eps010_trusted_sensitivity_cn.png", dpi=180)
    plt.close(fig)


def _record(branch_row: dict, audit_rows: list[dict], wall_s: float) -> str:
    """Render the Chinese experiment report."""
    table = [
        "| 方向 | 状态 | 相对误差 | 方向余弦 | 有限差分范数 | 预测范数 | 正向成功 | 负向成功 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in audit_rows:
        table.append(
            f"| {row['direction']} | {row['status']} | {row['vector_rel_err']:.3e} | "
            f"{row['vector_cosine']:.3e} | {row['fd_norm']:.3e} | {row['pred_norm']:.3e} | "
            f"{row['plus_success']} | {row['minus_success']} |"
        )
    passed = sum(row["status"] == "publishable" for row in audit_rows)
    max_err = max(row["vector_rel_err"] for row in audit_rows)
    return "\n".join(
        [
            "# P2c95 n2 受信任分支 eps=0.010 灵敏度复核记录",
            "",
            "## 实验目的",
            "",
            "P2c79 已证明 n2 受信任分支在 `eps=0.005` 下通过 9 个输入方向的扰动重求解一致性。P2c92 至 P2c94 又证明该分支在 100 微米末端目标邻域内可闭环跟踪。本轮将电流扰动尺度提高到 `eps=0.010`，复核局部线性化是否仍保持发布级误差。",
            "",
            "## 实验设置",
            "",
            "- 分支：n2 受信任分支。",
            "- 任务：末端三维位置。",
            f"- 电流扰动尺度：`{EPS:.3f}`。",
            "- 方向集合：3 个任务模态方向与 6 个固定随机方向。",
            f"- 发布门禁：线性化相对误差 `< {p2c79.TASK_GATE:.3f}`，正负扰动重求解均成功。",
            "",
            "## 基线门禁",
            "",
            f"- 基线是否通过：`{branch_row['baseline_ok']}`",
            f"- 末端位置误差：`{branch_row['position_error_m']:.3e} m`",
            f"- 一阶最优性：`{branch_row['stationarity_norm']:.3e}`",
            f"- 载荷传播比例：`{branch_row['force_ratio']:.3e}`",
            f"- 任务条件数：`{branch_row['task_condition']:.3e}`",
            "",
            "## 扰动重求解结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"共有 `{passed}` / `{len(audit_rows)}` 个方向通过 `eps=0.010` 扰动重求解一致性门禁，最大相对误差为 `{max_err:.3e}`。",
            "若全部通过，则 n2 受信任分支的末端三维位置局部灵敏度在 `eps=0.005` 与 `eps=0.010` 两个尺度上均具备发布级线性化证据。若部分失败，则应将发布尺度限制在更小扰动范围内，并沿失败方向做边界夹逼。",
            "",
            "该结论仍属于分支条件结论，不解除初始化敏感与多平衡分支风险。",
            "",
            "## 下一步计划",
            "",
            "1. 若全通过，尝试 `eps=0.0125` 或做电流方向随机批量复核。",
            "2. 若失败，沿失败方向做扰动尺度二分。",
            "3. 暂不推进完整末端位姿灵敏度。",
            "",
            f"- 本轮耗时：`{wall_s:.2f} s`",
            "",
        ]
    )


def main() -> None:
    p2c79.OUT_DIR = OUT_DIR
    p2c79.EPS = EPS
    t0 = time.perf_counter()
    params, I0 = p2c79.p2c57.p2c24._params_from_saved_case(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA)
    target_p, y_slack = p2c79.p2c57.p2c24._load_saved_target_and_seed(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA)
    y_seed = p2c79.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c79.p2c57.Args.steps_per_flex)
    branch = p2c79._recover_branch(params, I0, target_p, y_seed, "trusted_seed")
    branch_row = p2c79._branch_row(branch, branch)
    audit_rows = [p2c79._audit_direction(branch, I0, name, direction) for name, direction in p2c79._direction_set(branch)]
    wall_s = float(time.perf_counter() - t0)
    p2c79._write_csv(OUT_DIR / "branch_rows.csv", [branch_row])
    p2c79._write_csv(OUT_DIR / "random_direction_rows.csv", audit_rows)
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
        "wall_s": wall_s,
        "branch_row": branch_row,
        "audit_rows": audit_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(p2c79._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(audit_rows)
    (OUT_DIR / "record.md").write_text(_record(branch_row, audit_rows, wall_s), encoding="utf-8")
    print(json.dumps(p2c79._jsonable({"wall_s": wall_s, "failures": failures}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
