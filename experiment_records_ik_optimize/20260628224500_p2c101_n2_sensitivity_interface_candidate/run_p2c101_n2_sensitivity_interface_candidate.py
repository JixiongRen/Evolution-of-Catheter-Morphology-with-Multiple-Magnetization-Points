"""Validate the n2 branch-conditioned sensitivity interface candidate."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

from n2_branch_conditioned_interface import N2BranchConditionedSensitivityInterface


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _record(snapshot: dict) -> str:
    """Render the Chinese interface report."""
    decision = snapshot["decision"]
    evidence = snapshot["evidence"]
    return "\n".join(
        [
            "# P2c101 n2 分支条件灵敏度接口候选记录",
            "",
            "## 实验目的",
            "",
            "本轮不重新求解，而是把 P2c97、P2c98 和 P2c100 的证据整理为一个只读接口候选。目标是明确后续若要工程化使用 n2 灵敏度，应暴露哪些信息，以及哪些使用方式必须禁止。",
            "",
            "## 接口语义",
            "",
            "- 输入：已提交的 P2c97、P2c98 与 P2c100 证据文件。",
            "- 输出：工程判定、通过门禁数量、最大线性化误差、目标闭环范围、是否允许作为外层 IK 默认灵敏度。",
            "- 限定：该接口只代表受信任分支条件下的三维末端位置局部导数，不输出完整末端位姿灵敏度。",
            "",
            "## 验证结果",
            "",
            f"- 对象：`{decision['object_name']}`",
            f"- 任务：`{decision['task_name']}`",
            f"- 状态：`{decision['status']}`",
            f"- 随机电流方向通过：`{evidence['random_current_passed']}` / `{evidence['random_current_total']}`",
            f"- 最大线性化相对误差：`{evidence['random_current_max_rel_err']:.3e}`",
            f"- 目标闭环实验组通过：`{evidence['target_loop_passed']}` / `{evidence['target_loop_total']}`",
            f"- 最大电流扰动尺度：`{evidence['eps_max']:.4f}`",
            f"- 目标闭环半径：`{evidence['target_radius_m']:.3e} m`",
            f"- 是否允许作为外层 IK 默认灵敏度：`{snapshot['outer_ik_default_allowed']}`",
            "",
            "## 结论",
            "",
            "接口候选能够把当前证据链压缩为明确的工程决策：n2 三维末端位置灵敏度是分支条件发布候选，但不能作为初始化无关或默认生产灵敏度。该接口适合作为后续工程化封装的输入输出规范。",
            "",
            "## 下一步计划",
            "",
            "1. 若进入代码工程化，应将该只读证据接口扩展为实际计算接口，并保留所有硬门禁检查。",
            "2. 在接入外层 IK 前，必须增加可靠分支选择或分支恢复机制。",
            "3. 同步记录耗时，避免接口封装掩盖批量复核成本。",
            "",
        ]
    )


def main() -> None:
    interface = N2BranchConditionedSensitivityInterface(ROOT)
    snapshot = interface.as_dict()
    rows = [
        {
            "object": snapshot["decision"]["object_name"],
            "task": snapshot["decision"]["task_name"],
            "status": snapshot["decision"]["status"],
            "random_current_passed": snapshot["evidence"]["random_current_passed"],
            "random_current_total": snapshot["evidence"]["random_current_total"],
            "random_current_max_rel_err": snapshot["evidence"]["random_current_max_rel_err"],
            "target_loop_passed": snapshot["evidence"]["target_loop_passed"],
            "target_loop_total": snapshot["evidence"]["target_loop_total"],
            "outer_ik_default_allowed": snapshot["outer_ik_default_allowed"],
        }
    ]
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(snapshot), encoding="utf-8")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
