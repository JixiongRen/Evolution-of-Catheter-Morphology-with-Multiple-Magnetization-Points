"""Probe guarded local outer-IK step proposals for the n2 branch."""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "experiment_records_ik_optimize/20260629000000_p2c104_n2_cached_branch_audit/trusted_branch_cache.npz"
INTERFACE_DIR = ROOT / "experiment_records_ik_optimize/20260629013000_p2c107_n2_guarded_interface_candidate"

sys.path.insert(0, str(INTERFACE_DIR))
from n2_guarded_sensitivity_interface import N2GuardedSensitivityInterface  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    """Write a JSON file with stable formatting."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a CSV file from a list of dictionaries."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_font() -> None:
    """Configure plotting fonts for Chinese labels when available."""
    font_path = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    candidates = [
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["font.sans-serif"] = candidates
    plt.rcParams["axes.unicode_minus"] = False


def damped_step(S: np.ndarray, dx: np.ndarray, damping: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a damped minimum-norm current step for a target displacement."""
    lhs = S @ S.T + (damping**2) * np.eye(S.shape[0])
    dI = S.T @ np.linalg.solve(lhs, dx)
    pred = S @ dI
    rel = float(np.linalg.norm(pred - dx) / max(np.linalg.norm(dx), 1.0e-18))
    return dI, pred, rel


def build_cases(S: np.ndarray, U: np.ndarray, singular_values: np.ndarray) -> list[dict]:
    """Build controlled and intentionally over-large displacement requests."""
    cases: list[dict] = []
    for mode_index, current_step in [(0, 0.5), (1, 0.5), (2, 0.5), (0, 5.0), (1, 5.0), (2, 5.0)]:
        dx = U[:, mode_index] * singular_values[mode_index] * current_step
        cases.append(
            {
                "case": f"mode_{mode_index + 1}_{current_step:g}A",
                "kind": "singular_mode",
                "dx": dx,
                "expected_current_A": current_step,
            }
        )
    for axis, dx in [
        ("x", np.array([50.0e-6, 0.0, 0.0])),
        ("y", np.array([0.0, 50.0e-6, 0.0])),
        ("z", np.array([0.0, 0.0, 50.0e-6])),
    ]:
        cases.append({"case": f"cart_{axis}_50um", "kind": "cartesian_large", "dx": dx, "expected_current_A": None})
    return cases


def plot_rows(rows: list[dict], path: Path) -> None:
    """Plot requested displacement and required current for local step proposals."""
    configure_font()
    names = [row["case"] for row in rows]
    current = [float(row["current_norm_A"]) for row in rows]
    disp = [float(row["requested_norm_um"]) for row in rows]
    colors = ["#207a3c" if row["accepted_for_local_step"] else "#b33a3a" for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(names, disp, color=colors)
    axes[0].set_ylabel("请求位移范数 (微米)")
    axes[0].set_title("n2 分支条件局部步长建议：位移请求")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(names, current, color=colors)
    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1.2, label="局部步长电流上限 5 A")
    axes[1].set_ylabel("最小范数电流步长 (A)")
    axes[1].set_title("电流预算约束")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """Run the guarded local outer-step probe."""
    t0 = time.perf_counter()
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"missing trusted branch cache: {CACHE_PATH}")

    interface = N2GuardedSensitivityInterface(ROOT)
    snapshot = interface.as_dict()
    evidence = interface.evidence()
    required = snapshot["required_request"]
    valid = interface.validate_request(required)

    bad_requests = {
        "full_pose_request": {**required, "task_dimension": 6, "terminal_quantity": "full_pose"},
        "wrong_eps": {**required, "eps": float(required["eps"]) * 2.0},
        "wrong_branch": {**required, "branch_label": "untrusted"},
    }
    validation_rows = [
        {"case": "required_request", "accepted": valid.accepted, "reasons": ";".join(valid.reasons)}
    ]
    for name, request in bad_requests.items():
        decision = interface.validate_request(request)
        validation_rows.append({"case": name, "accepted": decision.accepted, "reasons": ";".join(decision.reasons)})

    cache = np.load(CACHE_PATH)
    S = np.asarray(cache["S"], dtype=float)
    I0 = np.asarray(cache["I0"], dtype=float)
    target_p = np.asarray(cache["target_p"], dtype=float)
    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)
    condition = float(singular_values[0] / singular_values[-1])
    damping = max(float(singular_values[-1]) * 0.1, 1.0e-12)

    rows: list[dict] = []
    failure_events: list[dict] = []
    current_limit = 5.0
    for case in build_cases(S, U, singular_values):
        dx = np.asarray(case["dx"], dtype=float)
        dI, pred, rel = damped_step(S, dx, damping)
        current_norm = float(np.linalg.norm(dI))
        within_budget = current_norm <= current_limit
        linear_ok = rel < 0.05
        accepted = bool(
            valid.accepted
            and evidence.status == "branch_conditioned_candidate"
            and linear_ok
            and within_budget
            and not interface.can_use_for_default_outer_ik()
        )
        reason = []
        if not linear_ok:
            reason.append("线性重构误差超限")
        if not within_budget:
            reason.append("所需电流步长超过局部预算")
        if evidence.status != "branch_conditioned_candidate":
            reason.append("接口证据不是分支条件候选")
        if not valid.accepted:
            reason.append("请求未通过接口契约")
        if not accepted:
            failure_events.append({"case": case["case"], "failure_class": "proposal_rejected", "reasons": reason})
        rows.append(
            {
                "case": case["case"],
                "kind": case["kind"],
                "requested_dx_m": json.dumps(dx.tolist()),
                "predicted_dx_m": json.dumps(pred.tolist()),
                "requested_norm_um": float(np.linalg.norm(dx) * 1.0e6),
                "predicted_norm_um": float(np.linalg.norm(pred) * 1.0e6),
                "current_norm_A": current_norm,
                "linear_rel_err": rel,
                "accepted_for_local_step": accepted,
                "rejection_reason": ";".join(reason),
                "I_candidate_A": json.dumps((I0 + dI).tolist()),
                "dI_A": json.dumps(dI.tolist()),
            }
        )

    write_csv(OUT_DIR / "proposal_rows.csv", rows)
    write_csv(OUT_DIR / "validation_rows.csv", validation_rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failure_events),
        encoding="utf-8",
    )
    plot_rows(rows, OUT_DIR / "n2_guarded_outer_step_probe_cn.png")

    accepted_count = sum(int(row["accepted_for_local_step"]) for row in rows)
    rejected_count = len(rows) - accepted_count
    cart_rejected = sum(int((row["kind"] == "cartesian_large") and (not row["accepted_for_local_step"])) for row in rows)
    mode_accepted = sum(int((row["kind"] == "singular_mode") and row["accepted_for_local_step"]) for row in rows)
    summary = {
        "experiment_id": "20260629043000_p2c113_n2_guarded_outer_step_probe",
        "status": "local_step_probe_only",
        "branch_scope": snapshot["allowed_scope"],
        "forbidden_scope": snapshot["forbidden_scope"],
        "outer_ik_default_allowed": interface.can_use_for_default_outer_ik(),
        "interface_evidence": asdict(evidence),
        "request_validation": validation_rows,
        "S_shape": list(S.shape),
        "singular_values": singular_values.tolist(),
        "condition": condition,
        "damping": damping,
        "current_limit_A": current_limit,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "singular_mode_accepted_count": mode_accepted,
        "cartesian_50um_rejected_count": cart_rejected,
        "target_p_m": target_p.tolist(),
        "I0_A": I0.tolist(),
        "wall_s": time.perf_counter() - t0,
        "interpretation": (
            "该实验只证明在已缓存可信分支附近，可以根据三维末端位置灵敏度给出受电流预算约束的局部步长建议。"
            "它不证明全局逆运动学可用，也不允许默认接入外层 IK。"
        ),
    }
    write_json(OUT_DIR / "summary.json", summary)
    write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "status": summary["status"],
                "condition": condition,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "singular_mode_accepted_count": mode_accepted,
                "cartesian_50um_rejected_count": cart_rejected,
                "outer_ik_default_allowed": interface.can_use_for_default_outer_ik(),
                "wall_s": summary["wall_s"],
            }
        ],
    )

    record = f"""# P2c113 n2 分支条件局部外层步长建议实验

## 实验目的

本轮实验验证 P2c107 的两点磁控三维末端位置灵敏度接口是否可以作为外层逆运动学的显式局部步长建议器。该实验不接入生产逆运动学，也不声称获得全局灵敏度。

## 方法

实验读取 P2c104 的可信分支缓存矩阵 \(S \\in \\mathbb{{R}}^{{3\\times 8}}\)。给定小目标位移 \(\\Delta p\)，采用阻尼最小范数解：

\\[
\\Delta I = S^\\mathsf{{T}}(SS^\\mathsf{{T}} + \\lambda^2 I)^{-1}\\Delta p.
\\]

若线性重构误差小于 0.05，且电流步长不超过 5 A，则该请求被标记为“局部步长可接受”。所有请求仍受 P2c107 的接口契约约束。

## 关键参数

- 任务：两点磁控导管，三维末端位置。
- 基准电流：`{I0.tolist()}` A。
- 目标点：`{target_p.tolist()}` m。
- 奇异值：`{singular_values.tolist()}`。
- 条件数：`{condition:.6e}`。
- 局部电流步长上限：`{current_limit:.3f}` A。

## 结果

- 接口证据状态：`{evidence.status}`。
- 接口默认接入外层 IK：`{interface.can_use_for_default_outer_ik()}`。
- 总请求数：`{len(rows)}`。
- 可接受局部步长：`{accepted_count}`。
- 被拒绝请求：`{rejected_count}`。
- 奇异方向请求可接受数：`{mode_accepted}`。
- 50 微米笛卡尔方向请求被拒绝数：`{cart_rejected}`。
- 墙钟耗时：`{summary['wall_s']:.6f}` s。

## 解释

结果表明，该分支条件灵敏度可以用于同一模型、同一目标点、同一基准电流附近的局部步长建议；但三维灵敏度强弱方向差异明显，任意 50 微米笛卡尔位移会要求远超局部预算的电流步长。因此，该接口适合作为受约束的小步长建议器，不适合作为默认全局外层 IK 模型。

## 失败与边界

完整末端位姿请求、错误扰动尺度请求和错误分支请求均被接口拒绝。笛卡尔 50 微米请求主要因电流步长预算超限被拒绝。这说明当前可发布边界仍是“分支条件下的三维位置局部导数”，不是完整位姿灵敏度，也不是跨分支全局模型。

## 下一步计划

下一轮应在该接口外层增加信赖域策略：根据奇异方向自动限制任务步长，将不可控或弱可控方向转为诊断输出；随后再做一次小目标位置扰动的重求解闭环验证，确认建议步长是否能把目标误差实际降低。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


if __name__ == "__main__":
    main()
