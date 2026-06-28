"""Trust-region step filtering for the guarded n2 position sensitivity."""

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
    """Write stable JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_font() -> None:
    """Configure fonts for Chinese plots."""
    font_path = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def minimum_norm_step(S: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the minimum-norm current step for one displacement request."""
    dI = np.linalg.pinv(S, rcond=1.0e-12) @ dx
    pred = S @ dI
    rel = float(np.linalg.norm(pred - dx) / max(np.linalg.norm(dx), 1.0e-18))
    return dI, pred, rel


def classify_step(dx: np.ndarray, S: np.ndarray, current_limit: float, min_scale: float) -> dict:
    """Classify a requested task displacement under a current trust region."""
    dI_full, pred_full, rel_full = minimum_norm_step(S, dx)
    full_current = float(np.linalg.norm(dI_full))
    if full_current <= current_limit:
        scale = 1.0
        action = "完整执行"
    else:
        scale = float(current_limit / max(full_current, 1.0e-18))
        action = "缩放执行" if scale >= min_scale else "仅诊断"
    dI = dI_full * scale
    pred = S @ dI
    remaining = dx - pred
    reason = ""
    if action == "缩放执行":
        reason = "完整请求超过电流预算，但缩放后仍有有效进展"
    elif action == "仅诊断":
        reason = "完整请求远超局部电流预算，缩放比例过小"
    return {
        "action": action,
        "reason": reason,
        "scale": scale,
        "dI_full": dI_full,
        "dI": dI,
        "pred_full": pred_full,
        "pred": pred,
        "full_current": full_current,
        "current": float(np.linalg.norm(dI)),
        "linear_rel_err_full": rel_full,
        "remaining": remaining,
    }


def build_cases(U: np.ndarray, singular_values: np.ndarray) -> list[dict]:
    """Build representative target displacement requests."""
    rows: list[dict] = []
    for mode_index, displacement_um in [(0, 5.0), (1, 0.25), (2, 0.02), (0, 25.0), (1, 2.0), (2, 0.2)]:
        dx = U[:, mode_index] * displacement_um * 1.0e-6
        rows.append(
            {
                "case": f"mode_{mode_index + 1}_{displacement_um:g}um",
                "kind": "principal_direction",
                "dx": dx,
                "nominal_current_A": float((displacement_um * 1.0e-6) / singular_values[mode_index]),
            }
        )
    for axis, displacement_um in [
        ("x", 1.0),
        ("y", 1.0),
        ("z", 1.0),
        ("x", 10.0),
        ("y", 10.0),
        ("z", 10.0),
        ("x", 50.0),
        ("y", 50.0),
        ("z", 50.0),
    ]:
        dx = np.zeros(3)
        dx[{"x": 0, "y": 1, "z": 2}[axis]] = displacement_um * 1.0e-6
        rows.append({"case": f"cart_{axis}_{displacement_um:g}um", "kind": "cartesian", "dx": dx, "nominal_current_A": None})
    rng = np.random.default_rng(20260629)
    for i in range(12):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        magnitude_um = [1.0, 5.0, 10.0, 25.0][i % 4]
        rows.append({"case": f"random_{i:02d}_{magnitude_um:g}um", "kind": "random", "dx": direction * magnitude_um * 1.0e-6, "nominal_current_A": None})
    return rows


def plot_rows(rows: list[dict], path: Path) -> None:
    """Plot trust-region action and scaling summary."""
    configure_font()
    names = [row["case"] for row in rows]
    scales = [float(row["scale"]) for row in rows]
    currents = [float(row["full_current_A"]) for row in rows]
    color_map = {"完整执行": "#207a3c", "缩放执行": "#c58a19", "仅诊断": "#b33a3a"}
    colors = [color_map[row["action"]] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].bar(names, scales, color=colors)
    axes[0].axhline(1.0, color="black", linewidth=1.0)
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="最小可执行比例 0.05")
    axes[0].set_ylabel("建议执行比例")
    axes[0].set_title("n2 三维位置请求的信赖域筛选")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(names, currents, color=colors)
    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="电流步长预算 5 A")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("完整请求所需电流 (A, 对数坐标)")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """Run the trust-region filtering probe."""
    t0 = time.perf_counter()
    interface = N2GuardedSensitivityInterface(ROOT)
    evidence = interface.evidence()
    snapshot = interface.as_dict()
    request_ok = interface.validate_request(snapshot["required_request"]).accepted
    cache = np.load(CACHE_PATH)
    S = np.asarray(cache["S"], dtype=float)
    I0 = np.asarray(cache["I0"], dtype=float)
    U, singular_values, _ = np.linalg.svd(S, full_matrices=False)
    current_limit = 5.0
    min_scale = 0.05

    rows: list[dict] = []
    failure_events: list[dict] = []
    for case in build_cases(U, singular_values):
        dx = np.asarray(case["dx"], dtype=float)
        result = classify_step(dx, S, current_limit, min_scale)
        if result["action"] == "仅诊断":
            failure_events.append(
                {
                    "case": case["case"],
                    "failure_class": "trust_region_too_small",
                    "reason": result["reason"],
                    "scale": result["scale"],
                    "full_current_A": result["full_current"],
                }
            )
        rows.append(
            {
                "case": case["case"],
                "kind": case["kind"],
                "action": result["action"],
                "reason": result["reason"],
                "requested_norm_um": float(np.linalg.norm(dx) * 1.0e6),
                "predicted_step_norm_um": float(np.linalg.norm(result["pred"]) * 1.0e6),
                "remaining_norm_um": float(np.linalg.norm(result["remaining"]) * 1.0e6),
                "scale": result["scale"],
                "full_current_A": result["full_current"],
                "suggested_current_A": result["current"],
                "linear_rel_err_full": result["linear_rel_err_full"],
                "requested_dx_m": json.dumps(dx.tolist()),
                "suggested_dx_m": json.dumps(result["pred"].tolist()),
                "dI_A": json.dumps(result["dI"].tolist()),
                "I_candidate_A": json.dumps((I0 + result["dI"]).tolist()),
            }
        )

    action_counts = {key: sum(int(row["action"] == key) for row in rows) for key in ["完整执行", "缩放执行", "仅诊断"]}
    summary = {
        "experiment_id": "20260629050000_p2c114_n2_trust_region_step_filter",
        "status": "trust_region_filter_probe",
        "interface_status": evidence.status,
        "request_contract_accepted": request_ok,
        "outer_ik_default_allowed": interface.can_use_for_default_outer_ik(),
        "interface_evidence": asdict(evidence),
        "current_limit_A": current_limit,
        "min_executable_scale": min_scale,
        "singular_values": singular_values.tolist(),
        "condition": float(singular_values[0] / singular_values[-1]),
        "action_counts": action_counts,
        "total_cases": len(rows),
        "wall_s": time.perf_counter() - t0,
        "interpretation": (
            "该实验将 P2c113 的局部步长建议扩展为信赖域筛选策略。"
            "它可以阻止弱可控方向的大位移请求直接进入外层 IK，但仍未替代扰动重求解验证。"
        ),
    }
    write_csv(OUT_DIR / "trust_region_rows.csv", rows)
    write_csv(OUT_DIR / "summary.csv", [{**action_counts, "wall_s": summary["wall_s"], "condition": summary["condition"]}])
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failure_events),
        encoding="utf-8",
    )
    plot_rows(rows, OUT_DIR / "n2_trust_region_step_filter_cn.png")

    record = f"""# P2c114 n2 三维位置灵敏度信赖域筛选实验

## 实验目的

P2c113 表明，两点磁控导管在可信分支附近已有可复核的三维末端位置灵敏度，但任意笛卡尔位移请求可能需要不现实的电流步长。本轮实验建立外层逆运动学使用前的信赖域筛选规则：先判断完整请求是否在电流预算内；若不在预算内，则计算可执行比例；若比例过小，则只作为诊断信息，不建议执行。

## 方法

设局部线性模型为

\\[
\\Delta p = S \\Delta I .
\\]

对给定位移请求 \\(\\Delta p\\)，先求最小范数电流步长 \\(\\Delta I_0 = S^+ \\Delta p\\)。若 \\(\\|\\Delta I_0\\| \\leq 5\\,\\mathrm{{A}}\\)，则完整执行；否则按比例

\\[
\\alpha = \\frac{{5}}{{\\|\\Delta I_0\\|}}
\\]

缩放位移与电流步长。若 \\(\\alpha < 0.05\\)，说明该请求相对当前灵敏度过大或方向弱可控，标记为仅诊断。

## 参数

- 任务口径：两点磁控导管，三维末端位置。
- 电流步长预算：5 A。
- 最小可执行比例：0.05。
- 奇异值：`{singular_values.tolist()}`。
- 条件数：`{float(singular_values[0] / singular_values[-1]):.6e}`。
- 接口状态：`{evidence.status}`。
- 默认接入外层 IK：`{interface.can_use_for_default_outer_ik()}`。

## 结果

- 完整执行：{action_counts["完整执行"]} 组。
- 缩放执行：{action_counts["缩放执行"]} 组。
- 仅诊断：{action_counts["仅诊断"]} 组。
- 总样本数：{len(rows)} 组。
- 墙钟耗时：{summary["wall_s"]:.6f} s。

## 分析

信赖域筛选能够把外层请求分成三类：第一类可按局部灵敏度完整执行；第二类方向上可控但幅值过大，应缩小步长；第三类需要电流过大，说明该方向在当前工作点局部不可作为有效控制方向。该策略没有降低物理门禁，也没有把局部导数伪装成全局模型。

## 工程边界

该结果仍只适用于同一可信分支、同一基准电流和同一任务口径。它不能替代扰动重求解审计，不能发布完整末端位姿灵敏度，也不能默认接入生产外层 IK。

## 下一步计划

下一轮应选择“完整执行”和“缩放执行”样本，进行小规模扰动重求解闭环：检查建议电流步长是否实际降低末端位置误差，并记录是否出现分支漂移。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


if __name__ == "__main__":
    main()
