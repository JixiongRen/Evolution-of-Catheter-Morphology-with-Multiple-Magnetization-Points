"""Diagnose A1/A2 supervised IK failures from existing formal runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import yaml


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
FORMAL_ROOT = ROOT / "formal_experiments"
DIAG_ROOT = ROOT / "formal_diagnostics"
CONFIG_PATH = ROOT / "catheter_physical_configs.yaml"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

SOURCE_RUN_DEFAULT = FORMAL_ROOT / "20260706114453_A1_A2_5mm_supervised_IK_no_gravity"
REPRESENTATIVE_POINTS = {
    "A1": [1, 8, 12, 13, 17],
    "A2": [5, 10, 9, 17, 19],
}


def _font() -> FontProperties:
    if FONT_PATH.exists():
        return FontProperties(fname=str(FONT_PATH))
    return FontProperties()


def _configure_fonts() -> None:
    if FONT_PATH.exists():
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _vec(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full((3,), np.nan, dtype=float)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _jax_device_info() -> dict[str, Any]:
    try:
        import jax

        devices = [str(dev) for dev in jax.devices()]
        platforms = sorted({getattr(dev, "platform", "unknown") for dev in jax.devices()})
        return {"available": True, "devices": devices, "platforms": platforms, "has_gpu": "gpu" in platforms}
    except Exception as exc:
        return {"available": False, "error": repr(exc), "devices": [], "platforms": [], "has_gpu": False}


def _load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _expected_moment(config: dict[str, Any], catheter_type: str) -> list[float]:
    return list(config["catheter_types"][catheter_type]["solver_input"]["m_body_list_A_m2"][0])


def _extract_command_args(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    marker = "COMMAND_ARGS:"
    marker2 = "STDOUT_STDERR:"
    if marker not in text or marker2 not in text:
        return {"raw_args": [], "parse_ok": False}
    block = text.split(marker, 1)[1].split(marker2, 1)[0].strip()
    try:
        args = json.loads(block)
    except Exception:
        return {"raw_args": [], "parse_ok": False}
    return {"raw_args": args, "parse_ok": True}


def _arg_value(args: list[str], name: str) -> str | None:
    if name in args:
        idx = args.index(name)
        if idx + 1 < len(args):
            return str(args[idx + 1])
    prefix = name + "="
    for item in args:
        if str(item).startswith(prefix):
            return str(item)[len(prefix) :]
    return None


def _parse_m_body(value: str | None) -> list[list[float]]:
    if not value:
        return []
    rows = []
    for part in str(value).split(";"):
        part = part.strip()
        if not part:
            continue
        rows.append([float(x.strip()) for x in part.split(",")])
    return rows


def _log_environment_flags(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return {
        "cpu_fallback": "Falling back to cpu" in text or "cpu" in text.lower() and "cuda" in text.lower(),
        "cuda_error": "xla_cuda" in text or "CUDA" in text or "cuInit" in text,
        "auto_timeout": "[AUTO_TIMEOUT]" in text,
        "traceback": "Traceback" in text,
    }


def _point_dir(source_run: Path, catheter_type: str, index: int) -> Path:
    return source_run / catheter_type / "formal_20pt" / f"pt_{index:02d}"


def _centerline_path(point_dir: Path, run_config: dict[str, Any]) -> Path | None:
    for section in ("final", "last_accept", "init"):
        rel = run_config.get(section, {}).get("centerline_npy")
        if rel:
            path = point_dir / rel
            if path.exists():
                return path
    poses = sorted((point_dir / "poses" / "accept").glob("centerline_iter_*.npy"))
    return poses[-1] if poses else None


def _endpoint_row(
    *,
    source_run: Path,
    config: dict[str, Any],
    summary_row: dict[str, str],
) -> dict[str, Any]:
    catheter = summary_row["catheter_type"]
    idx = int(summary_row["target_index"])
    point = _point_dir(source_run, catheter, idx)
    result = _read_json(point / "result.json", {})
    history = _read_json(point / "history.json", {"outer": []})
    run_config = _read_json(point / "run_config.json", {})
    log_path = Path(summary_row.get("log_path") or point.with_suffix(".log"))
    if not log_path.exists():
        log_path = source_run / catheter / "formal_20pt" / f"pt_{idx:02d}.log"

    env = _log_environment_flags(log_path)
    command = _extract_command_args(log_path)
    raw_args = command["raw_args"]
    m_body = _parse_m_body(_arg_value(raw_args, "--m_body_list"))
    expected_m = _expected_moment(config, catheter)
    p_des = _vec(result.get("p_des", [math.nan, math.nan, math.nan]))
    p_flex = _vec(result.get("p_tip_flex", result.get("p_tip", [math.nan, math.nan, math.nan])))
    p_true = _vec(result.get("p_tip_true", [math.nan, math.nan, math.nan]))

    cl_path = _centerline_path(point, run_config)
    cl_end = np.full((3,), np.nan, dtype=float)
    cl_points = 0
    if cl_path is not None:
        try:
            cl = np.load(cl_path)
            cl_points = int(cl.shape[0])
            cl_end = np.asarray(cl[-1], dtype=float).reshape(3)
        except Exception:
            pass

    d_cl_flex = float(np.linalg.norm(cl_end - p_flex) * 1000.0)
    d_cl_true = float(np.linalg.norm(cl_end - p_true) * 1000.0)
    d_flex_true = float(np.linalg.norm(p_flex - p_true) * 1000.0)
    err_flex = float(np.linalg.norm(p_flex - p_des) * 1000.0)
    err_true = float(np.linalg.norm(p_true - p_des) * 1000.0)

    convention = "unknown"
    if np.isfinite(d_cl_true) and d_cl_true < 0.25:
        convention = "centerline_matches_reported_true_tip"
    elif np.isfinite(d_cl_flex) and d_cl_flex < 0.25:
        convention = "centerline_matches_flex_tip"
    elif np.isfinite(d_cl_true) and 0.8 <= d_cl_true <= 2.2 and np.isfinite(d_cl_flex) and d_cl_flex > d_cl_true:
        convention = "reported_true_tip_likely_shifted_beyond_centerline_end"
    elif np.isfinite(d_cl_flex) and np.isfinite(d_cl_true):
        convention = "centerline_endpoint_conflicts_with_reported_tips"

    result_success = _to_bool(result.get("success"))
    flex_success = _to_bool(result.get("success_flex_eps"))
    true_success = _to_bool(result.get("success_true_eps"))
    objective_mismatch = bool(result_success == flex_success and result_success != true_success)

    outer = list(history.get("outer", []))
    best_true = min([_to_float(item.get("err_true_mm")) for item in outer] or [math.nan])
    best_flex = min([_to_float(item.get("err_flex_mm", item.get("err_mm"))) for item in outer] or [math.nan])
    accepted = sum(1 for item in outer if bool(item.get("accept", False)))
    rejected = sum(1 for item in outer if str(item.get("status", "")).upper() == "REJECT")
    last = outer[-1] if outer else {}

    m_body_ok = bool(len(m_body) == 1 and np.allclose(np.asarray(m_body[0], dtype=float), np.asarray(expected_m, dtype=float)))
    single_segment_ok = str(_arg_value(raw_args, "--M_list") or "") == "5" and len(m_body) == 1
    gpu_valid = not bool(env["cpu_fallback"] or env["cuda_error"])

    return {
        "catheter_type": catheter,
        "target_index": idx,
        "theta_rad": _to_float(summary_row.get("theta_rad")),
        "representative": idx in REPRESENTATIVE_POINTS.get(catheter, []),
        "run_dir": str(point.resolve()),
        "log_path": str(log_path.resolve()),
        "gpu_valid_from_log": gpu_valid,
        "cpu_fallback_or_cuda_error": bool(env["cpu_fallback"] or env["cuda_error"]),
        "auto_timeout": bool(env["auto_timeout"]),
        "traceback_in_log": bool(env["traceback"]),
        "m_body_ok": m_body_ok,
        "single_segment_ok": single_segment_ok,
        "expected_m_body": json.dumps(expected_m),
        "actual_m_body": json.dumps(m_body),
        "err_flex_mm": err_flex,
        "err_true_mm": err_true,
        "best_err_flex_mm": best_flex,
        "best_err_true_mm": best_true,
        "result_success": result_success,
        "success_flex_eps": flex_success,
        "success_true_eps": true_success,
        "objective_success_mismatch": objective_mismatch,
        "centerline_npy": str(cl_path.resolve()) if cl_path else "",
        "centerline_points": cl_points,
        "centerline_end_x_m": float(cl_end[0]),
        "centerline_end_y_m": float(cl_end[1]),
        "centerline_end_z_m": float(cl_end[2]),
        "centerline_end_to_flex_tip_mm": d_cl_flex,
        "centerline_end_to_reported_true_tip_mm": d_cl_true,
        "flex_tip_to_reported_true_tip_mm": d_flex_true,
        "terminal_convention_class": convention,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "outer_history_count": len(outer),
        "last_lambda": _to_float(last.get("lam")),
        "last_rho": _to_float(last.get("rho")),
        "last_pred_red": _to_float(last.get("pred_red")),
        "last_act_red": _to_float(last.get("act_red")),
        "last_dI_max_abs": _to_float(last.get("dI_max_abs")),
        "last_dx_abs_m": _to_float(last.get("dx_abs_m")),
        "summary_stop_reason": summary_row.get("stop_reason", ""),
        "summary_wall_s": _to_float(summary_row.get("wall_s")),
        "summary_cls": summary_row.get("cls", ""),
        "summary_normE": _to_float(summary_row.get("normE")),
    }


def _trust_rows(source_run: Path, summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for srow in summary_rows:
        catheter = srow["catheter_type"]
        idx = int(srow["target_index"])
        point = _point_dir(source_run, catheter, idx)
        history = _read_json(point / "history.json", {"outer": []})
        for item in history.get("outer", []):
            pred = _to_float(item.get("pred_red"))
            act = _to_float(item.get("act_red"))
            rho = _to_float(item.get("rho"))
            rows.append(
                {
                    "catheter_type": catheter,
                    "target_index": idx,
                    "representative": idx in REPRESENTATIVE_POINTS.get(catheter, []),
                    "iter": item.get("iter"),
                    "status": item.get("status"),
                    "accept": bool(item.get("accept", False)),
                    "err_flex_mm": _to_float(item.get("err_flex_mm", item.get("err_mm"))),
                    "err_true_mm": _to_float(item.get("err_true_mm")),
                    "lambda": _to_float(item.get("lam")),
                    "rho": rho,
                    "pred_red": pred,
                    "act_red": act,
                    "rho_recomputed": act / max(pred, 1e-18) if math.isfinite(pred) and math.isfinite(act) else math.nan,
                    "prediction_sign_wrong": bool(math.isfinite(pred) and pred > 0.0 and math.isfinite(act) and act < 0.0),
                    "tiny_step": bool(_to_float(item.get("dI_max_abs")) < 1e-4 and _to_float(item.get("dx_abs_m")) < 1e-7),
                    "dI_max_abs_A": _to_float(item.get("dI_max_abs")),
                    "dx_abs_m": _to_float(item.get("dx_abs_m")),
                    "x_m": _to_float(item.get("x_m")),
                    "cls": item.get("cls"),
                    "normE": _to_float(item.get("normE")),
                    "sensitivity_status": (item.get("sensitivity") or {}).get("status"),
                    "stationarity_norm": _to_float((item.get("sensitivity") or {}).get("stationarity_norm")),
                }
            )
    return rows


def _implicit_sensitivity_rows(source_run: Path, summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for srow in summary_rows:
        catheter = srow["catheter_type"]
        idx = int(srow["target_index"])
        point = _point_dir(source_run, catheter, idx)
        history = _read_json(point / "history.json", {"outer": []})
        for item in history.get("outer", []):
            sens = item.get("sensitivity") or {}
            if not sens:
                continue
            rows.append(
                {
                    "catheter_type": catheter,
                    "target_index": idx,
                    "representative": idx in REPRESENTATIVE_POINTS.get(catheter, []),
                    "iter": item.get("iter"),
                    "status": item.get("status"),
                    "accept": bool(item.get("accept", False)),
                    "sensitivity_status": sens.get("status"),
                    "x_method": sens.get("x_method"),
                    "method": sens.get("method"),
                    "stationarity_norm": _to_float(sens.get("stationarity_norm")),
                    "condition_number": _to_float(sens.get("condition_number")),
                    "err_true_mm": _to_float(item.get("err_true_mm")),
                    "err_flex_mm": _to_float(item.get("err_flex_mm", item.get("err_mm"))),
                    "rho": _to_float(item.get("rho")),
                }
            )
    return rows


def _classify_root_cause(endpoint: dict[str, Any], trust_point_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if endpoint["cpu_fallback_or_cuda_error"]:
        return "环境无效", "日志显示 CUDA 初始化失败或回退 CPU，该点不能用于耗时和分支稳定性结论。"
    if endpoint["auto_timeout"]:
        return "运行未完整结束", "该点触发自动超时，现有记录不能证明自然终止后的算法行为。"
    if endpoint["objective_success_mismatch"]:
        return "目标函数与判据不一致", "求解器按柔性段末端判断成功，但正式报告按最后刚段远端统计成功。"
    if endpoint["terminal_convention_class"] != "centerline_matches_reported_true_tip":
        return "末端几何约定不清", "中心线终点、柔性段末端和报告的真实末端互不一致，需先统一物理末端定义。"

    final_err = float(endpoint["err_true_mm"])
    if final_err <= 5.0:
        return "有效成功", "真实末端误差已进入 5 mm 阈值。"

    if trust_point_rows:
        last_rows = trust_point_rows[-5:]
        tiny_reject = sum(1 for row in last_rows if row["tiny_step"] and not row["accept"])
        sign_wrong = sum(1 for row in trust_point_rows if row["prediction_sign_wrong"])
        max_lam = max([_to_float(row.get("lambda")) for row in trust_point_rows] or [math.nan])
        if tiny_reject >= 2 or max_lam >= 1e5:
            return "阻尼过大导致停滞", "后期步长接近零且阻尼急剧增大，外层局部模型已失效。"
        if sign_wrong >= 3:
            return "局部预测方向错误", "多次出现预测下降但实际目标函数上升，需审计灵敏度和步长策略。"

    if endpoint["best_err_true_mm"] <= 10.0:
        return "边界收敛", "历史最好误差已接近目标阈值，优先检查步长、初值和末端定义。"
    if str(endpoint.get("summary_cls", "")) != "GOOD":
        return "内层平衡质量不足", "最终 FK 平衡未达到 GOOD，末端误差和灵敏度都可能受到弱收敛状态污染。"
    if final_err > 5.0:
        return "外层优化未达目标", "最终 FK 平衡为 GOOD 但真实末端误差仍远大于阈值，优先检查灵敏度方向、步长接受策略和初值分支。"
    return "待可达性判断", "现有证据不足以区分优化失败和目标疑似不可达。"


def _evidence_and_confidence(cause: str, endpoint: dict[str, Any], trust_point_rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Assign evidence strength, confidence, and conclusion type for a point."""
    if cause in {"环境无效", "运行未完整结束", "目标函数与判据不一致", "末端几何约定不清"}:
        return "强", "高", "当前阻断项"
    if cause == "有效成功":
        return "强", "高", "已排除因素"
    if cause == "内层平衡质量不足":
        return "中", "中", "已确认根因"
    if cause == "阻尼过大导致停滞":
        last_rows = trust_point_rows[-5:]
        tiny_or_high_lam = [
            row
            for row in last_rows
            if bool(row.get("tiny_step")) or _to_float(row.get("lambda")) >= 1e5
        ]
        if len(tiny_or_high_lam) >= 2:
            return "强", "高", "已确认根因"
        return "中", "中", "仅为伴随现象或放大因素"
    if cause == "局部预测方向错误":
        sign_wrong = sum(1 for row in trust_point_rows if bool(row.get("prediction_sign_wrong")))
        if sign_wrong >= 3:
            return "强", "高", "已确认根因"
        return "中", "中", "仍未判明但已有下一步证据路径"
    if cause == "边界收敛":
        return "中", "中", "仍未判明但已有下一步证据路径"
    if cause == "外层优化未达目标":
        cls_good = str(endpoint.get("summary_cls", "")) == "GOOD"
        if cls_good and float(endpoint.get("err_true_mm", math.inf)) > 20.0:
            return "中", "中", "仍未判明但已有下一步证据路径"
        return "弱", "低", "仍未判明但已有下一步证据路径"
    return "弱", "低", "仍未判明但已有下一步证据路径"


def _diagnostic_rows(endpoint_rows: list[dict[str, Any]], trust_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trust_rows:
        by_point[(row["catheter_type"], int(row["target_index"]))].append(row)

    out: list[dict[str, Any]] = []
    for endpoint in endpoint_rows:
        key = (endpoint["catheter_type"], int(endpoint["target_index"]))
        point_trust_rows = by_point.get(key, [])
        cause, evidence = _classify_root_cause(endpoint, point_trust_rows)
        evidence_level, confidence_level, conclusion_type = _evidence_and_confidence(cause, endpoint, point_trust_rows)
        gate_valid = bool(
            endpoint["gpu_valid_from_log"]
            and endpoint["m_body_ok"]
            and endpoint["single_segment_ok"]
            and endpoint["terminal_convention_class"] == "centerline_matches_reported_true_tip"
            and not endpoint["objective_success_mismatch"]
        )
        out.append(
            {
                **endpoint,
                "d0_gate_valid": gate_valid,
                "main_root_cause": cause,
                "evidence_level": evidence_level,
                "confidence_level": confidence_level,
                "conclusion_type": conclusion_type,
                "evidence_summary_zh": evidence,
                "next_action_zh": _next_action(cause),
            }
        )
    return out


def _next_action(cause: str) -> str:
    table = {
        "环境无效": "修复 GPU 环境或强制重跑；该点暂不进入算法统计。",
        "运行未完整结束": "增加非缓冲日志和周期性快照后重跑；当前仅用于定位超时阶段。",
        "目标函数与判据不一致": "把外层残差、终止条件和报告判据统一到同一个真实末端点。",
        "末端几何约定不清": "确认 packed rigid state 表示中点还是远端，并修正 true tip 计算函数。",
        "有效成功": "作为后续灵敏度审计的正常样本。",
        "阻尼过大导致停滞": "在末端约定修复后，加入自适应信赖域、回溯和重新线性化机制。",
        "局部预测方向错误": "执行中心差分灵敏度审计；若误差超过 10%，先修复隐式灵敏度。",
        "边界收敛": "进行多初值、反向遍历和局部加密可达性诊断。",
        "内层平衡质量不足": "先重跑该点或提高 FK 平衡可靠性，再审计外层灵敏度。",
        "外层优化未达目标": "优先执行中心差分灵敏度审计、局部预测审计和多初值分支诊断。",
        "待可达性判断": "进入经验可达图采样，不能直接判定不可达。",
    }
    return table.get(cause, "补充证据后再决定修改项。")


def _plot_error_by_angle(run_dir: Path, diag_rows: list[dict[str, Any]]) -> None:
    font = _font()
    for catheter in ("A1", "A2"):
        rows = [r for r in diag_rows if r["catheter_type"] == catheter]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: int(r["target_index"]))
        theta = np.asarray([float(r["theta_rad"]) for r in rows], dtype=float)
        err_true = np.asarray([float(r["err_true_mm"]) for r in rows], dtype=float)
        err_flex = np.asarray([float(r["err_flex_mm"]) for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
        ax.plot(theta, err_true, "o-", lw=2.0, label="最后刚段远端误差")
        ax.plot(theta, err_flex, "s--", lw=1.5, label="柔性段末端误差")
        ax.axhline(5.0, color="#d92525", ls="--", lw=1.2, label="5 mm 阈值")
        for r in rows:
            if r["representative"]:
                ax.text(float(r["theta_rad"]), float(r["err_true_mm"]), f"pt{int(r['target_index']):02d}", fontproperties=font, fontsize=8)
        ax.set_xlabel("圆周角度 θ (rad)", fontproperties=font)
        ax.set_ylabel("末端位置误差 (mm)", fontproperties=font)
        ax.set_title(f"{catheter} 末端误差随目标角度变化", fontproperties=font)
        ax.grid(True, alpha=0.3)
        ax.legend(prop=font)
        fig.tight_layout()
        fig.savefig(run_dir / f"{catheter}_error_by_angle.png", bbox_inches="tight")
        fig.savefig(run_dir / f"{catheter}_error_by_angle.svg", bbox_inches="tight")
        plt.close(fig)


def _plot_root_cause_distribution(run_dir: Path, diag_rows: list[dict[str, Any]]) -> None:
    font = _font()
    counts = Counter(str(row["main_root_cause"]) for row in diag_rows)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=180)
    ax.bar(np.arange(len(labels)), values, color="#3b6ea8")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontproperties=font, rotation=30, ha="right")
    ax.set_ylabel("点数", fontproperties=font)
    ax.set_title("A1/A2 失败主因分布", fontproperties=font)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "root_cause_distribution.png", bbox_inches="tight")
    fig.savefig(run_dir / "root_cause_distribution.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_endpoint_convention(run_dir: Path, endpoint_rows: list[dict[str, Any]]) -> None:
    font = _font()
    rows = [r for r in endpoint_rows if r["representative"]]
    rows = sorted(rows, key=lambda r: (r["catheter_type"], int(r["target_index"])))
    labels = [f"{r['catheter_type']}-pt{int(r['target_index']):02d}" for r in rows]
    d_flex = np.asarray([float(r["centerline_end_to_flex_tip_mm"]) for r in rows], dtype=float)
    d_true = np.asarray([float(r["centerline_end_to_reported_true_tip_mm"]) for r in rows], dtype=float)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=180)
    ax.bar(x - 0.18, d_flex, width=0.36, label="中心线终点到柔性段末端")
    ax.bar(x + 0.18, d_true, width=0.36, label="中心线终点到报告真实末端")
    ax.axhline(0.25, color="#d92525", ls="--", lw=1.0, label="几何一致参考线")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=font, rotation=30, ha="right")
    ax.set_ylabel("距离 (mm)", fontproperties=font)
    ax.set_title("代表点末端几何约定一致性检查", fontproperties=font)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(prop=font)
    fig.tight_layout()
    fig.savefig(run_dir / "endpoint_convention_representative.png", bbox_inches="tight")
    fig.savefig(run_dir / "endpoint_convention_representative.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_representative_traces(source_run: Path, run_dir: Path) -> None:
    font = _font()
    for catheter, indices in REPRESENTATIVE_POINTS.items():
        fig, axes = plt.subplots(len(indices), 1, figsize=(9.0, 2.6 * len(indices)), dpi=170, sharex=False)
        if len(indices) == 1:
            axes = [axes]
        for ax, idx in zip(axes, indices):
            point = _point_dir(source_run, catheter, idx)
            history = _read_json(point / "history.json", {"outer": []})
            outer = history.get("outer", [])
            if not outer:
                ax.set_title(f"{catheter}-pt{idx:02d}: 无历史记录", fontproperties=font)
                continue
            k = np.asarray([int(item.get("iter", i)) for i, item in enumerate(outer)], dtype=float)
            err_true = np.asarray([_to_float(item.get("err_true_mm")) for item in outer], dtype=float)
            err_flex = np.asarray([_to_float(item.get("err_flex_mm", item.get("err_mm"))) for item in outer], dtype=float)
            accepts = np.asarray([bool(item.get("accept", False)) for item in outer], dtype=bool)
            ax.plot(k, err_true, color="#d92525", lw=1.8, label="最后刚段远端")
            ax.plot(k, err_flex, color="#145cff", lw=1.4, ls="--", label="柔性段末端")
            ax.scatter(k[accepts], err_true[accepts], s=16, color="#168a2f", label="接受步" if idx == indices[0] else None)
            ax.axhline(5.0, color="#555555", ls=":", lw=1.0)
            ax.set_ylabel("误差 (mm)", fontproperties=font)
            ax.set_title(f"{catheter}-pt{idx:02d}", fontproperties=font, fontsize=10)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("外层迭代步", fontproperties=font)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, prop=font, loc="upper right")
        fig.suptitle(f"{catheter} 代表点收敛轨迹", fontproperties=font, fontsize=13)
        fig.tight_layout(rect=(0, 0, 0.92, 0.97))
        fig.savefig(run_dir / f"{catheter}_representative_error_traces.png", bbox_inches="tight")
        fig.savefig(run_dir / f"{catheter}_representative_error_traces.svg", bbox_inches="tight")
        plt.close(fig)


def _write_record(
    *,
    run_dir: Path,
    source_run: Path,
    env: dict[str, Any],
    diag_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
    trust_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
) -> None:
    reps = [r for r in diag_rows if r["representative"]]
    all_count = len(diag_rows)
    d0_valid_count = sum(1 for r in diag_rows if r["d0_gate_valid"])
    cpu_invalid = sum(1 for r in diag_rows if r["cpu_fallback_or_cuda_error"])
    objective_mismatch = sum(1 for r in diag_rows if r["objective_success_mismatch"])
    convention_conflict = sum(1 for r in diag_rows if r["terminal_convention_class"] != "centerline_matches_reported_true_tip")
    causes = Counter(str(r["main_root_cause"]) for r in diag_rows)
    rep_lines = []
    for row in reps:
        rep_lines.append(
            f"| {row['catheter_type']}-pt{int(row['target_index']):02d} | "
            f"{row['err_true_mm']:.3f} | {row['err_flex_mm']:.3f} | "
            f"{row['terminal_convention_class']} | {row['main_root_cause']} | "
            f"{row['evidence_level']} | {row['confidence_level']} | {row['conclusion_type']} | {row['evidence_summary_zh']} |"
        )

    success_count = sum(1 for r in diag_rows if float(r["err_true_mm"]) <= 5.0)
    good_large_error = sum(1 for r in diag_rows if str(r.get("summary_cls", "")) == "GOOD" and float(r["err_true_mm"]) > 5.0)
    weak_large_error = sum(1 for r in diag_rows if str(r.get("summary_cls", "")) != "GOOD" and float(r["err_true_mm"]) > 5.0)
    if d0_valid_count == all_count:
        gate_conclusion = (
            "本轮数据的 D0 前置门禁已通过：运行日志未显示 CPU 回退，磁化配置与单磁体结构一致，"
            "中心线终点与报告真实末端一致，求解器成功标志也已统一到真实末端。"
        )
    else:
        gate_conclusion = (
            "本轮数据尚不能直接进入完整灵敏度审计。需要先处理环境、磁化配置、末端定义或成功判据中的前置问题。"
        )

    if d0_valid_count == all_count:
        direct_constraints = [
            "1. 对误差较大的 GOOD 点，首要问题不是内层 FK 崩溃，而是外层控制更新未能把真实末端推向目标；下一步应优先审计灵敏度方向、局部预测下降量和步长接受机制。",
            "2. 对误差较大且 FK 只达到 WEAK 的点，必须先保证 FK 平衡质量，再判断 IK 是否失败；弱平衡状态下的灵敏度不应直接作为算法结论。",
            "3. 成功点可作为正常样本，用于中心差分灵敏度审计；失败点需要补做多初值、反向遍历和局部可达性采样，以区分优化器失败与疑似不可达。",
            "4. 若中心差分显示灵敏度误差超过 10%，优先修复隐式灵敏度；若小步预测准确但大步失败，则优先改信赖域和回溯策略。",
        ]
        next_suggestion = (
            "建议下一步只针对本轮代表点进入 D1/D2/D3：先做真实末端中心差分灵敏度审计，"
            "再做局部预测审计和多初值分支诊断；暂不扩大到全 40 点，避免在根因未定位前消耗大量算力。"
        )
        skipped_stage_note = "本轮已完成 D0 门禁和历史轨迹审计，D1/D3/D4 尚未执行，但已不再受 D0 阻断。"
    else:
        direct_constraints = [
            "1. 先统一物理末端定义。需要确认 `x_rigid_list_bar` 表示刚段中点还是刚段远端；若中心线提取逻辑正确，则当前 `_true_rigid_distal_tip()` 很可能把刚段远端又向前平移了半个刚段长度。",
            "2. 外层位置残差、自然终止条件、`result.json` 中的 `success` 字段、正式实验统计必须使用同一个末端点。否则会出现柔性段末端达标但真实末端未达标的假成功。",
            "3. GPU 回退点必须作废并重跑。现有部分 A1/A2 点的日志显示 CUDA 初始化失败，这些点不能用于耗时或分支稳定性结论。",
            "4. D1 中心差分灵敏度审计应在上述三项修复后再执行；否则差分对象和隐式灵敏度对象并不一致，误差结论没有物理意义。",
            "5. 对已存在的历史轨迹，阻尼过大和步长接近零的停滞现象确实存在，但它应作为第二优先级处理，排在末端定义和环境门禁之后。",
        ]
        next_suggestion = (
            "建议先执行一个小修复批次：统一末端定义、修正成功标志、在日志中强制记录 JAX 设备和工作空间越界信息。"
            "随后只重跑十个代表点，确认 D0 门禁通过，再开展 D1 灵敏度中心差分审计和 D3 多初值分支诊断。"
        )
        skipped_stage_note = "本轮因 D0 门禁未完全通过，D1/D3/D4 暂不执行。"

    lines = [
        "# A1/A2 有监督逆运动学失败诊断报告",
        "",
        "## 一、实验目的",
        "",
        "本报告执行失败诊断计划的前置门禁与历史轨迹审计，目标是解释 A1/A2 单点磁化导管在无重力、5 mm 阈值圆周任务中的失败来源，并把证据转换为后续代码修改任务。",
        "",
        "## 二、数据来源",
        "",
        f"- 源实验目录：`{source_run}`",
        f"- 诊断输出目录：`{run_dir}`",
        f"- git 版本：`{_git_commit()}`",
        f"- 当前 JAX 设备：`{env}`",
        "",
        "## 三、门禁结论",
        "",
        f"- 共检查 {all_count} 个点，其中 D0 门禁完全有效的点为 {d0_valid_count} 个。",
        f"- 日志中存在 CUDA 初始化失败或 CPU 回退的点数：{cpu_invalid}。",
        f"- 求解器成功标志与真实末端成功标志不一致的点数：{objective_mismatch}。",
        f"- 中心线终点、柔性段末端和报告真实末端存在几何约定冲突的点数：{convention_conflict}。",
        "",
        f"结论：{gate_conclusion}",
        "",
        f"- 真实末端进入 5 mm 阈值的点数：{success_count}/{all_count}。",
        f"- FK 平衡为 GOOD 但真实末端未达标的点数：{good_large_error}。",
        f"- FK 平衡非 GOOD 且真实末端未达标的点数：{weak_large_error}。",
        "",
        "## 四、代表点诊断",
        "",
        "| 点位 | 真实末端误差(mm) | 柔性末端误差(mm) | 末端几何分类 | 主因 | 证据等级 | 可信度 | 结论类型 | 证据 |",
        "|---|---:|---:|---|---|---|---|---|---|",
        *rep_lines,
        "",
        "## 五、全局主因分布",
        "",
        "| 主因 | 点数 |",
        "|---|---:|",
    ]
    for key, value in causes.most_common():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## 六、对后续算法修改的直接约束",
            "",
            *direct_constraints,
            "",
            "## 七、已生成文件",
            "",
            "- `diagnostic_summary.csv`：每个点的主因、证据和下一步任务。",
            "- `endpoint_convention_audit.csv`：末端几何约定和目标函数错配审计。",
            "- `trust_region_audit.csv`：历史预测下降、真实下降、阻尼和步长记录。",
            "- `implicit_sensitivity_log_summary.csv`：历史记录中的隐式灵敏度状态摘要。",
            f"- `sensitivity_columns.csv`：D1 中心差分灵敏度审计占位文件；{skipped_stage_note}",
            f"- `branch_trials.csv`：D3 多初值与分支诊断占位文件；{skipped_stage_note}",
            f"- `reachability_samples.npz`：D4 经验可达图占位文件；{skipped_stage_note}",
            "- `runtime_environment.json`：当前运行环境和设备信息。",
            "- `*_error_by_angle.png`：误差随圆周角度变化。",
            "- `endpoint_convention_representative.png`：代表点末端约定冲突图。",
            "- `root_cause_distribution.png`：主因分布图。",
            "",
            "## 八、下一步建议",
            "",
            next_suggestion,
        ]
    )
    (run_dir / "record.md").write_text("\n".join(lines), encoding="utf-8")


def _summarize_diagnostics(diag_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a machine-readable summary for diagnostic conclusions."""
    by_catheter: dict[str, Any] = {}
    for catheter in sorted({str(row["catheter_type"]) for row in diag_rows}):
        rows = [row for row in diag_rows if str(row["catheter_type"]) == catheter]
        by_catheter[catheter] = {
            "count": len(rows),
            "d0_valid_count": sum(1 for row in rows if bool(row["d0_gate_valid"])),
            "success_5mm_count": sum(1 for row in rows if float(row["err_true_mm"]) <= 5.0),
            "root_cause_counts": dict(Counter(str(row["main_root_cause"]) for row in rows)),
            "conclusion_type_counts": dict(Counter(str(row["conclusion_type"]) for row in rows)),
            "confidence_counts": dict(Counter(str(row["confidence_level"]) for row in rows)),
            "high_confidence_failures": [
                {
                    "target_index": int(row["target_index"]),
                    "err_true_mm": float(row["err_true_mm"]),
                    "main_root_cause": str(row["main_root_cause"]),
                    "next_action_zh": str(row["next_action_zh"]),
                }
                for row in rows
                if float(row["err_true_mm"]) > 5.0 and str(row["confidence_level"]) == "高"
            ],
        }
    blockers = [
        row
        for row in diag_rows
        if str(row.get("conclusion_type")) == "当前阻断项"
    ]
    return {
        "count": len(diag_rows),
        "d0_valid_count": sum(1 for row in diag_rows if bool(row["d0_gate_valid"])),
        "success_5mm_count": sum(1 for row in diag_rows if float(row["err_true_mm"]) <= 5.0),
        "root_cause_counts": dict(Counter(str(row["main_root_cause"]) for row in diag_rows)),
        "conclusion_type_counts": dict(Counter(str(row["conclusion_type"]) for row in diag_rows)),
        "confidence_counts": dict(Counter(str(row["confidence_level"]) for row in diag_rows)),
        "current_blocker_count": len(blockers),
        "current_blockers": [
            {
                "catheter_type": str(row["catheter_type"]),
                "target_index": int(row["target_index"]),
                "main_root_cause": str(row["main_root_cause"]),
                "evidence_summary_zh": str(row["evidence_summary_zh"]),
            }
            for row in blockers
        ],
        "by_catheter": by_catheter,
    }


def _write_failure_events(run_dir: Path, diag_rows: list[dict[str, Any]]) -> None:
    """Write one JSON event per non-successful target."""
    with (run_dir / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in diag_rows:
            if float(row["err_true_mm"]) <= 5.0:
                continue
            event = {
                "catheter_type": str(row["catheter_type"]),
                "target_index": int(row["target_index"]),
                "err_true_mm": float(row["err_true_mm"]),
                "fk_class": str(row.get("summary_cls", "")),
                "d0_gate_valid": bool(row["d0_gate_valid"]),
                "main_root_cause": str(row["main_root_cause"]),
                "evidence_level": str(row["evidence_level"]),
                "confidence_level": str(row["confidence_level"]),
                "conclusion_type": str(row["conclusion_type"]),
                "evidence_summary_zh": str(row["evidence_summary_zh"]),
                "next_action_zh": str(row["next_action_zh"]),
                "run_dir": str(row.get("run_dir", "")),
            }
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_diagnostics(source_run: Path, out_root: Path) -> Path:
    _configure_fonts()
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = out_root / f"{run_id}_A1_A2_failure_diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _read_csv_rows(source_run / "summary.csv")
    config = _load_config()
    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_run": str(source_run.resolve()),
        "git_commit": _git_commit(),
        "python": sys.executable,
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "jax": _jax_device_info(),
    }

    endpoint_rows = [_endpoint_row(source_run=source_run, config=config, summary_row=row) for row in summary_rows]
    trust = _trust_rows(source_run, summary_rows)
    sensitivity = _implicit_sensitivity_rows(source_run, summary_rows)
    diagnostics = _diagnostic_rows(endpoint_rows, trust)
    d0_valid_count = sum(1 for row in diagnostics if row["d0_gate_valid"])
    d0_blocked = d0_valid_count != len(diagnostics)
    if d0_blocked:
        sensitivity_reason = "末端几何约定、目标函数或环境门禁尚未完全通过，中心差分灵敏度审计暂不具备物理解释意义。"
        branch_reason = "末端几何约定或环境门禁未完全通过，暂不开展多初值和平衡分支诊断。"
        reachability_reason = "末端几何约定或环境门禁未完全通过，暂不开展经验可达图采样。"
    else:
        sensitivity_reason = "本轮已完成 D0 门禁和历史轨迹审计，尚未执行中心差分灵敏度审计；可作为下一阶段任务。"
        branch_reason = "本轮已完成 D0 门禁和历史轨迹审计，尚未执行多初值和平衡分支诊断；可作为下一阶段任务。"
        reachability_reason = "本轮已完成 D0 门禁和历史轨迹审计，尚未执行经验可达图采样；可作为下一阶段任务。"

    _write_csv(run_dir / "endpoint_convention_audit.csv", endpoint_rows)
    _write_csv(run_dir / "trust_region_audit.csv", trust)
    _write_csv(run_dir / "implicit_sensitivity_log_summary.csv", sensitivity)
    _write_csv(run_dir / "diagnostic_summary.csv", diagnostics)
    summary = {
        "metadata": env,
        "source_run": str(source_run.resolve()),
        "diagnostic_summary": _summarize_diagnostics(diagnostics),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_failure_events(run_dir, diagnostics)
    _write_csv(
        run_dir / "sensitivity_columns.csv",
        [
            {
                "status": "not_run",
                "blocked_by": "D0" if d0_blocked else "not_blocked",
                "reason_zh": sensitivity_reason,
            }
        ],
    )
    _write_csv(
        run_dir / "branch_trials.csv",
        [
            {
                "status": "not_run",
                "blocked_by": "D0" if d0_blocked else "not_blocked",
                "reason_zh": branch_reason,
            }
        ],
    )
    np.savez_compressed(
        run_dir / "reachability_samples.npz",
        status=np.asarray(["not_run"]),
        blocked_by=np.asarray(["D0" if d0_blocked else "not_blocked"]),
        reason_zh=np.asarray([reachability_reason]),
        controls=np.zeros((0, 9), dtype=float),
        terminal_points=np.zeros((0, 3), dtype=float),
    )
    (run_dir / "runtime_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"{sys.executable} {Path(__file__).resolve()} "
        f"--source-run {source_run.resolve()} "
        f"--out-root {out_root.resolve()}",
    ]
    (run_dir / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")

    _plot_error_by_angle(run_dir, diagnostics)
    _plot_root_cause_distribution(run_dir, diagnostics)
    _plot_endpoint_convention(run_dir, endpoint_rows)
    _plot_representative_traces(source_run, run_dir)
    _write_record(
        run_dir=run_dir,
        source_run=source_run,
        env=env,
        diag_rows=diagnostics,
        endpoint_rows=endpoint_rows,
        trust_rows=trust,
        sensitivity_rows=sensitivity,
    )
    return run_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN_DEFAULT, help="Formal A1/A2 run directory.")
    parser.add_argument("--out-root", type=Path, default=DIAG_ROOT, help="Diagnostic output root directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    source_run = args.source_run.resolve()
    if not source_run.exists():
        raise FileNotFoundError(f"source run does not exist: {source_run}")
    if not (source_run / "summary.csv").exists():
        raise FileNotFoundError(f"summary.csv missing in source run: {source_run}")
    t0 = time.perf_counter()
    run_dir = run_diagnostics(source_run, args.out_root.resolve())
    print(f"diagnostic_dir: {run_dir}", flush=True)
    print(f"record: {run_dir / 'record.md'}", flush=True)
    print(f"wall_s: {time.perf_counter() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()
