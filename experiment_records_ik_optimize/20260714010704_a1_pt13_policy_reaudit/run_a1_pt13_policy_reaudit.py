"""Run the A1 pt13 branch-preserving channel policy reaudit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python")
REAUDIT_SCRIPT = ROOT / "DELIVER_TO_EXP" / "run_a1_a2_sensitivity_reaudit.py"
SOURCE_POINT = (
    ROOT
    / "DELIVER_TO_EXP"
    / "formal_experiments"
    / "20260713232320_A1_A2_5mm_supervised_IK_no_gravity"
    / "A1"
    / "small_regression"
    / "pt_13"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _git_info() -> dict[str, str]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    status_all = run(["git", "status", "--short"])
    untracked_count = sum(1 for line in status_all.splitlines() if line.startswith("?? "))
    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "tracked_status_short": run(["git", "status", "--short", "--untracked-files=no"]),
        "untracked_count": str(untracked_count),
    }


def _source_state_summary() -> dict[str, Any]:
    manifest_path = SOURCE_POINT / "branch_states" / "branch_state_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    best = min((row for row in rows if row.get("accept")), key=lambda row: float(row.get("terminal_error_mm", float("inf"))))
    final = next(row for row in reversed(rows) if row.get("phase") == "final")
    return {
        "source_point": str(SOURCE_POINT.relative_to(ROOT)),
        "manifest_rows": len(rows),
        "best_eval_id": best.get("eval_id"),
        "best_npz_path": str((SOURCE_POINT / "branch_states" / best["npz_path"]).relative_to(ROOT)),
        "best_terminal_error_mm": float(best.get("terminal_error_mm")),
        "best_z_star_hash": best.get("z_star_hash"),
        "final_eval_id": final.get("eval_id"),
        "final_npz_path": str((SOURCE_POINT / "branch_states" / final["npz_path"]).relative_to(ROOT)),
        "final_terminal_error_mm": float(final.get("terminal_error_mm")),
        "final_z_star_hash": final.get("z_star_hash"),
        "best_equals_final_hash": bool(best.get("z_star_hash") == final.get("z_star_hash")),
    }


def _run_audit_case(
    *,
    selector: str,
    segments: int,
    current_steps: str,
    x_steps_mm: str,
    current_channels: str,
    modes: str,
    terminal_gate_mm: float,
    centerline_gate_mm: float,
    weak_displacement_mm: float,
) -> dict[str, Any]:
    label = f"{selector}_seg{segments}"
    out_dir = EXP_DIR / "audit_runs" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(REAUDIT_SCRIPT),
        str(SOURCE_POINT),
        "--out-dir",
        str(out_dir),
        "--selector",
        selector,
        "--current-steps-a",
        current_steps,
        "--x-steps-mm",
        x_steps_mm,
        "--current-channels",
        current_channels,
        "--sensitivity-modes",
        modes,
        "--solve-mode",
        "svd",
        "--ridge",
        "1e-10",
        "--svd-rcond",
        "1e-9",
        "--spectral-filter",
        "hard",
        "--spectral-damping",
        "0.0",
        "--roundtrip-segments",
        str(int(segments)),
        "--terminal-gate-mm",
        f"{terminal_gate_mm:.12g}",
        "--centerline-gate-mm",
        f"{centerline_gate_mm:.12g}",
        "--weak-displacement-mm",
        f"{weak_displacement_mm:.12g}",
    ]
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_PLATFORMS"] = "cuda"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (out_dir / "stderr.log").open(
        "w", encoding="utf-8"
    ) as stderr:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "selector": selector,
        "roundtrip_segments": int(segments),
        "out_dir": str(out_dir.relative_to(ROOT)),
        "exit_code": int(proc.returncode),
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": float(elapsed),
        "command": " ".join(cmd),
    }


def _aggregate_outputs(run_rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fd_rows: list[dict[str, str]] = []
    sens_rows: list[dict[str, str]] = []
    for run in run_rows:
        out_dir = ROOT / str(run["out_dir"])
        for row in _read_csv(out_dir / "finite_difference_audit.csv"):
            row["audit_run"] = str(run["label"])
            fd_rows.append(row)
        for row in _read_csv(out_dir / "sensitivity_modes.csv"):
            row["audit_run"] = str(run["label"])
            sens_rows.append(row)
    _write_csv(EXP_DIR / "channel_audit_full.csv", fd_rows)
    _write_csv(EXP_DIR / "state_replay_gate.csv", sens_rows)
    return fd_rows, sens_rows


def _classify_group(rows: list[dict[str, str]]) -> dict[str, Any]:
    used_errors = [
        value
        for value in (_float_or_none(row.get("relative_error")) for row in rows if _truthy(row.get("used_in_stats")))
        if value is not None
    ]
    used_steps = [
        value for value in (_float_or_none(row.get("step")) for row in rows if _truthy(row.get("used_in_stats"))) if value is not None
    ]
    branch_rows = [row for row in rows if _truthy(row.get("branch_consistent"))]
    weak_rows = [row for row in rows if _truthy(row.get("weak_channel"))]
    if not used_errors:
        if weak_rows:
            classification = "弱响应"
        elif branch_rows:
            classification = "同分支误差超限"
        else:
            classification = "跨分支或不可审计"
        median_rel = None
        p90_rel = None
    else:
        median_rel = float(median(used_errors))
        p90_rel = _percentile(used_errors, 0.9)
        if median_rel <= 0.05 and (p90_rel is not None and p90_rel <= 0.10):
            classification = "可生产使用"
        elif median_rel <= 0.10 and (p90_rel is not None and p90_rel <= 0.15):
            classification = "可有限使用"
        else:
            classification = "同分支误差超限"
    return {
        "classification": classification,
        "row_count": len(rows),
        "used_row_count": len(used_errors),
        "branch_consistent_count": len(branch_rows),
        "weak_count": len(weak_rows),
        "median_relative_error": "" if median_rel is None else median_rel,
        "p90_relative_error": "" if p90_rel is None else p90_rel,
        "safe_step_abs": "" if not used_steps else max(used_steps),
        "min_step_abs": "" if not used_steps else min(used_steps),
        "max_fd_displacement_mm": max(
            [v for v in (_float_or_none(row.get("fd_displacement_mm")) for row in rows) if v is not None] or [0.0]
        ),
    }


def _build_policy(fd_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in fd_rows:
        key = (
            row.get("selector", ""),
            row.get("roundtrip_segments", ""),
            row.get("kind", ""),
            row.get("channel", ""),
            row.get("mode", ""),
        )
        grouped[key].append(row)
    policy_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        selector, segments, kind, channel, mode = key
        stats = _classify_group(rows)
        policy_rows.append(
            {
                "point_key": "A1_pt13",
                "selector": selector,
                "roundtrip_segments": segments,
                "kind": kind,
                "channel": channel,
                "mode": mode,
                **stats,
                "evidence": EXP_DIR.name,
            }
        )
    _write_csv(EXP_DIR / "channel_policy_candidate.csv", policy_rows)
    return policy_rows


def _build_decision_summary(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a conservative per-channel decision table from residual-space sensitivity."""

    rows = [row for row in policy_rows if str(row.get("mode")) == "residual_svd"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("kind")), str(row.get("channel")))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (kind, channel), items in sorted(grouped.items()):
        class_counts = Counter(str(row.get("classification")) for row in items)
        finite_rows = [row for row in items if str(row.get("classification")) == "可有限使用"]
        production_rows = [row for row in items if str(row.get("classification")) == "可生产使用"]
        if production_rows and not class_counts.get("同分支误差超限"):
            decision = "生产候选"
        elif finite_rows:
            decision = "有限使用候选"
        elif class_counts.get("跨分支或不可审计"):
            decision = "不可审计"
        else:
            decision = "暂不放行"
        numeric_errors = [
            float(row["median_relative_error"])
            for row in items
            if str(row.get("median_relative_error", "")).strip() not in ("", "nan")
        ]
        finite_steps = [
            float(row["safe_step_abs"])
            for row in finite_rows
            if str(row.get("safe_step_abs", "")).strip() not in ("", "nan")
        ]
        summary_rows.append(
            {
                "point_key": "A1_pt13",
                "kind": kind,
                "channel": channel,
                "mode": "residual_svd",
                "decision": decision,
                "classification_counts": json.dumps(dict(class_counts), ensure_ascii=False, sort_keys=True),
                "candidate_count": len(items),
                "limited_count": len(finite_rows),
                "production_count": len(production_rows),
                "best_median_relative_error": "" if not numeric_errors else min(numeric_errors),
                "limited_safe_step_abs_max": "" if not finite_steps else max(finite_steps),
                "limited_safe_step_abs_min": "" if not finite_steps else min(finite_steps),
            }
        )
    _write_csv(EXP_DIR / "channel_decision_summary.csv", summary_rows)
    return summary_rows


def _plot_policy(policy_rows: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt

        from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

        configure_matplotlib_cjk()
    except Exception as exc:
        return repr(exc)
    counts = Counter(str(row["classification"]) for row in policy_rows)
    labels = list(counts)
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values, color="#457b9d")
    ax.set_ylabel("通道-模式组合数量")
    ax.set_title("A1-pt13 同分支通道复审分类")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "channel_policy_summary.png", dpi=200)
    fig.savefig(EXP_DIR / "channel_policy_summary.svg")
    plt.close(fig)
    return None


def _write_record(summary: dict[str, Any]) -> None:
    decision_lines = []
    for row in summary.get("channel_decision_summary", []):
        name = "推进量" if row["kind"] == "x" else f"I{row['channel']}"
        best = row.get("best_median_relative_error", "")
        best_text = "无有效统计" if best == "" else f"{float(best) * 100.0:.2f}%"
        decision_lines.append(
            f"| {name} | {row['decision']} | {best_text} | {row['classification_counts']} |"
        )
    decision_table = "\n".join(
        ["| 通道 | 保守决策 | 最小中位相对误差 | 分类计数 |", "|---|---|---:|---|"] + decision_lines
    )
    text = f"""# A1-pt13 同分支通道策略复审

## 实验目的

本轮用于推进 A1/A2 有监督逆运动学修正主线中的最小闭环：先固定 A1-pt13 的近解分支，重审八路线圈电流和推进量通道，再生成点级策略候选表。该实验不改变正运动学物理模型、网格、容差、真实末端定义和控制边界。

## 实验设置

- 导管：A1。
- 点位：pt13。
- 重力：关闭。
- 离散化：`M_list=5`。
- 正运动学容差：`1e-5`。
- 成功目标：最后刚性段远端真实末端误差小于 `5 mm`。
- 源目录：`{summary['source_state']['source_point']}`。
- 源近解误差：{summary['source_state']['final_terminal_error_mm']:.6f} mm。
- 源状态哈希：`{summary['source_state']['final_z_star_hash']}`。
- 运行模式：`{summary['mode']}`。

## 输出文件

- `channel_audit_full.csv`：逐通道有限差分与灵敏度对比。
- `state_replay_gate.csv`：基准状态重载门禁与灵敏度求解统计。
- `channel_policy_candidate.csv`：按通道、状态选择器、回环分段和灵敏度模式汇总的候选策略表。
- `audit_summary.json`：机器可读摘要。
- `channel_policy_summary.png/svg`：中文分类图。

## 初步结果

- 子任务数量：{summary['run_count']}。
- 成功退出子任务：{summary['successful_run_count']}。
- 有限差分记录行数：{summary['fd_row_count']}。
- 灵敏度记录行数：{summary['sensitivity_row_count']}。
- 策略候选行数：{summary['policy_row_count']}。
- 分类统计：{summary['classification_counts']}。

## 通道决策摘要

以下表格只使用残差空间灵敏度结果进行保守聚合。它用于指导下一轮消融实验，不等同于最终交付结论。

{decision_table}

## 当前结论

若本轮为 `smoke` 模式，只能证明脚本链路可运行，不能替代全量复审结论。若本轮为 `full` 模式，当前结果显示：A1-pt13 没有任何通道达到生产使用标准；I5 仅达到有限使用候选；推进量和其他电流通道在残差空间灵敏度下仍应暂不放行。

## 下一步计划

下一步应运行 I5-only、x-only、I5+x 三臂同分支精修消融。由于推进量未通过灵敏度门禁，x-only 和 I5+x 只能作为经验消融，不应直接作为可信局部线性更新策略。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def _write_commands() -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda {PYTHON} \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_a1_pt13_policy_reaudit.py --mode full
"""
    path = EXP_DIR / "commands.sh"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--summarize-existing", action="store_true", help="Regenerate summaries from existing audit outputs.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    _write_commands()
    source_state = _source_state_summary()
    if args.summarize_existing:
        run_rows = _read_csv(EXP_DIR / "audit_run_status.csv")
        selectors = sorted({str(row.get("selector", "")) for row in run_rows if row.get("selector")})
        segments = sorted({int(row.get("roundtrip_segments", 0)) for row in run_rows if row.get("roundtrip_segments")})
        current_steps = "existing"
        x_steps_mm = "existing"
        current_channels = "existing"
        modes = "existing"
    elif args.mode == "smoke":
        selectors = ["final"]
        segments = [1]
        current_steps = "0.03125"
        x_steps_mm = "0.0625"
        current_channels = "5"
        modes = "residual_svd"
    else:
        selectors = ["final", "best_accept", "last_accept"]
        segments = [1, 4, 8]
        current_steps = "0.125,0.0625,0.03125,0.015625"
        x_steps_mm = "0.25,0.125,0.0625"
        current_channels = "0,1,2,3,4,5,6,7"
        modes = "solver_response,gn_svd,stationarity_exact,residual_svd"

    if not args.summarize_existing:
        run_rows = []
        for selector in selectors:
            for segment_count in segments:
                run_rows.append(
                    _run_audit_case(
                        selector=selector,
                        segments=segment_count,
                        current_steps=current_steps,
                        x_steps_mm=x_steps_mm,
                        current_channels=current_channels,
                        modes=modes,
                        terminal_gate_mm=0.1,
                        centerline_gate_mm=0.5,
                        weak_displacement_mm=0.05,
                    )
                )
        _write_csv(EXP_DIR / "audit_run_status.csv", run_rows)
    fd_rows, sens_rows = _aggregate_outputs(run_rows)
    policy_rows = _build_policy(fd_rows)
    decision_rows = _build_decision_summary(policy_rows)
    plot_error = _plot_policy(policy_rows)
    summary = {
        "experiment_id": EXP_DIR.name,
        "mode": args.mode,
        "git": _git_info(),
        "source_state": source_state,
        "selectors": selectors,
        "roundtrip_segments": segments,
        "current_steps_a": current_steps,
        "x_steps_mm": x_steps_mm,
        "current_channels": current_channels,
        "sensitivity_modes": modes,
        "run_count": len(run_rows),
        "successful_run_count": sum(1 for row in run_rows if int(row["exit_code"]) == 0),
        "fd_row_count": len(fd_rows),
        "sensitivity_row_count": len(sens_rows),
        "policy_row_count": len(policy_rows),
        "classification_counts": dict(Counter(str(row["classification"]) for row in policy_rows)),
        "channel_decision_summary": decision_rows,
        "plot_error": plot_error,
        "runs": run_rows,
    }
    _write_json(EXP_DIR / "audit_summary.json", summary)
    _write_record(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(int(row["exit_code"]) != 0 for row in run_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
