"""Audit endpoint-load closure for the n1 rigid magnet segment."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
P2C84_DIR = ROOT / "experiment_records_ik_optimize" / "20260628121600_p2c84_block_stationarity_load_closure_audit"
for path in (ROOT, P2C84_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p2c84_block_stationarity_load_closure_audit as p2c84


jax.config.update("jax_enable_x64", True)
OUT_DIR = Path(__file__).resolve().parent


def _set_chinese_font() -> None:
    """Use a local CJK font for plots."""
    for font_file in [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            prop = fm.FontProperties(fname=str(font_file))
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _jsonable(value: Any) -> Any:
    """Convert values into JSON-safe objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _cross_matrix_term(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Return r cross f."""
    return np.cross(np.asarray(r, dtype=float), np.asarray(f, dtype=float))


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / max(np.linalg.norm(q), 1.0e-30)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _recover_n1_case() -> dict[str, Any]:
    """Recover the P2c84 n1 near-gate branch."""
    spec = next(item for item in p2c84.CASES if item["label"] == "n1_soft_near_gate")
    return p2c84._solve_case(spec)


def _audit_interfaces(case: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Audit force and torque closure at each rigid-to-flexible interface."""
    params = case["params"]
    y = np.asarray(case["solve"]["y"], dtype=float)
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    x = p2c84.p2c57.p2c47.initial_state(params, blocks[0])
    force_rows: list[dict[str, Any]] = []
    torque_rows: list[dict[str, Any]] = []

    for i in range(len(params.flex) - 1):
        x = x.at[7:10].set(blocks[i, 0:3] * f_ref)
        x = x.at[10:13].set(blocks[i, 3:6] * m_ref)
        x = p2c84.p2c57.p2c47.normalize_state(x)
        x, _flex_points = p2c84.p2c57.p2c47._integrate_flex(params, x, i, p2c84.p2c57.Args.steps_per_flex)
        x_prox = np.asarray(jax.device_get(x), dtype=float)
        f_total, tau_total = p2c84.p2c57.p2c47._rigid_wrench(params, x, i)
        f_total_np = np.asarray(jax.device_get(f_total), dtype=float)
        tau_total_np = np.asarray(jax.device_get(tau_total), dtype=float)
        rigid = params.rigid[i]
        q = np.asarray(x_prox[3:7], dtype=float)
        q = q / max(np.linalg.norm(q), 1.0e-30)
        r_axis = np.asarray(_quat_to_rotmat_wxyz(q) @ np.asarray(jax.device_get(rigid.v_star), dtype=float), dtype=float)
        r_dist = float(rigid.length) * r_axis
        lever_term = _cross_matrix_term(r_dist, f_total_np)
        x_dist, _mid_p, _mid_q = p2c84.p2c57.p2c47._advance_rigid(params, x, i)
        x_dist_np = np.asarray(jax.device_get(x_dist), dtype=float)
        next_force = np.asarray(jax.device_get(blocks[i + 1, 0:3] * f_ref), dtype=float)
        next_torque = np.asarray(jax.device_get(blocks[i + 1, 3:6] * m_ref), dtype=float)
        force_jump = next_force - x_dist_np[7:10]
        current_torque_jump = next_torque - x_dist_np[10:13]
        pre_force = x_prox[7:10]
        pre_torque = x_prox[10:13]
        torque_candidates = {
            "近端集中载荷": pre_torque - tau_total_np - _cross_matrix_term(r_dist, pre_force),
            "中点等效载荷": x_dist_np[10:13],
            "远端集中载荷": pre_torque - tau_total_np - _cross_matrix_term(r_dist, pre_force) + lever_term,
        }
        force_row = {
            "interface": i,
            "rigid_length_m": float(rigid.length),
            "rigid_force_norm_N": float(np.linalg.norm(f_total_np)),
            "pre_force_norm_N": float(np.linalg.norm(pre_force)),
            "post_force_norm_N": float(np.linalg.norm(x_dist_np[7:10])),
            "next_force_norm_N": float(np.linalg.norm(next_force)),
            "force_jump_norm_N": float(np.linalg.norm(force_jump)),
            "force_jump_over_rigid_force": float(np.linalg.norm(force_jump) / max(np.linalg.norm(f_total_np), 1.0e-30)),
            "force_jump_over_internal_force": float(
                np.linalg.norm(force_jump)
                / max(np.linalg.norm(pre_force), np.linalg.norm(x_dist_np[7:10]), np.linalg.norm(next_force), 1.0e-30)
            ),
            "force_closure_publishable": bool(np.linalg.norm(force_jump) / max(np.linalg.norm(f_total_np), 1.0e-30) <= 1.5),
        }
        force_rows.append(force_row)
        for model_name, tau_post in torque_candidates.items():
            jump = next_torque - tau_post
            torque_rows.append(
                {
                    "interface": i,
                    "model": model_name,
                    "rigid_torque_norm_Nm": float(np.linalg.norm(tau_total_np)),
                    "lever_torque_norm_Nm": float(np.linalg.norm(lever_term)),
                    "post_torque_norm_Nm": float(np.linalg.norm(tau_post)),
                    "next_torque_norm_Nm": float(np.linalg.norm(next_torque)),
                    "torque_jump_norm_Nm": float(np.linalg.norm(jump)),
                    "torque_jump_over_rigid_torque": float(np.linalg.norm(jump) / max(np.linalg.norm(tau_total_np), 1.0e-30)),
                }
            )
        x = x_dist

    gate = p2c84._gate_row(case)
    summary = {
        "gate": gate,
        "max_force_jump_over_rigid_force": max((row["force_jump_over_rigid_force"] for row in force_rows), default=0.0),
        "best_torque_model": min(torque_rows, key=lambda row: row["torque_jump_over_rigid_torque"]) if torque_rows else None,
        "endpoint_load_can_fix_force_gate": False,
    }
    return force_rows, torque_rows, summary


def _plot(force_rows: list[dict[str, Any]], torque_rows: list[dict[str, Any]]) -> None:
    """Plot endpoint-load audit results."""
    _set_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    force_labels = [f"接口 {row['interface']}" for row in force_rows]
    force_vals = [row["force_jump_over_rigid_force"] for row in force_rows]
    axes[0].bar(force_labels, force_vals, color="#e76f51")
    axes[0].axhline(1.5, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("力连续性门禁")
    axes[0].set_ylabel("力跳变 / 刚段外力")
    axes[0].grid(axis="y", alpha=0.25)
    torque_labels = [row["model"] for row in torque_rows]
    torque_vals = [row["torque_jump_over_rigid_torque"] for row in torque_rows]
    axes[1].bar(torque_labels, torque_vals, color="#457b9d")
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("力矩传播模型对比")
    axes[1].set_ylabel("力矩跳变 / 刚段外矩")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c87 单点刚性磁体段端点载荷闭合审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_rigid_endpoint_load_closure_cn.png", dpi=180)
    plt.close(fig)


def _record(force_rows: list[dict[str, Any]], torque_rows: list[dict[str, Any]], summary: dict[str, Any], wall_s: float) -> str:
    """Render the Chinese experiment report."""
    force_table = [
        "| 接口 | 刚段外力/N | 力跳变/N | 力跳变比 | 内力归一跳变 | 是否过力门禁 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in force_rows:
        force_table.append(
            f"| {row['interface']} | {row['rigid_force_norm_N']:.3e} | {row['force_jump_norm_N']:.3e} | "
            f"{row['force_jump_over_rigid_force']:.3e} | {row['force_jump_over_internal_force']:.3e} | "
            f"{'是' if row['force_closure_publishable'] else '否'} |"
        )
    torque_table = [
        "| 力矩模型 | 刚段外矩/Nm | 力臂项/Nm | 力矩跳变/Nm | 力矩跳变比 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in torque_rows:
        torque_table.append(
            f"| {row['model']} | {row['rigid_torque_norm_Nm']:.3e} | {row['lever_torque_norm_Nm']:.3e} | "
            f"{row['torque_jump_norm_Nm']:.3e} | {row['torque_jump_over_rigid_torque']:.3e} |"
        )
    best = summary["best_torque_model"]
    return "\n".join(
        [
            "# P2c87 n1 刚性磁体段端点载荷闭合审计",
            "",
            "## 实验目的",
            "",
            "P2c86 表明，单点 n1 的不可发布性主要与末端位置约束、自由边界和刚段接口载荷传播之间的相容性有关。本轮进一步审计刚性磁体段载荷作用点假设：比较近端集中载荷、中点等效载荷和远端集中载荷三种力矩传播模型，判断端点载荷修正能否解释或修复当前载荷传播门禁失败。",
            "",
            "## 方法",
            "",
            "复现 P2c84 的 `n1_soft_near_gate` 分支，保持原求解参数不变。沿求解后的分段内力状态积分，在每个刚性磁体段接口处记录：柔性段末端内力、刚段外力、刚段传播后的内力、下一柔性段初始内力，并计算力跳变比例。随后只改变外力等效作用点，比较三种力矩传播公式下的力矩跳变。该审计不改变物理求解器，不发布灵敏度。",
            "",
            "## 原始门禁",
            "",
            f"- 一阶最优性：`{summary['gate']['stationarity_norm']:.3e}`",
            f"- 条件数：`{summary['gate']['jacobian_condition']:.3e}`",
            f"- 接口连续：`{summary['gate']['continuity_norm']:.3e}`",
            f"- 力跳变相对刚段外力：`{summary['gate']['max_interface_force_jump_over_rigid_force']:.3e}`",
            f"- 失败分类：`{summary['gate']['failure_class']}`",
            "",
            "## 力连续性结果",
            "",
            *force_table,
            "",
            "## 力矩传播模型结果",
            "",
            *torque_table,
            "",
            "## 结论",
            "",
            f"最优力矩模型为 `{best['model']}`，其力矩跳变比为 `{best['torque_jump_over_rigid_torque']:.3e}`。但是，端点/中点/近端载荷位置只影响力矩传播，不改变力连续性；当前 n1 的力跳变比仍为 `{summary['max_force_jump_over_rigid_force']:.3e}`，显著高于 1.5 门限。因此，单纯把刚段磁力从中心等效改为端点等效，不能使该单点样本进入可发布状态。",
            "",
            "这说明 n1 的主要闭合缺口更接近“刚段外力与相邻柔性段内力链不相容”，而不是单纯的外力作用点力矩误差。下一步若继续修复单点模型，应显式重构接口力连续残差或允许物理可解释的刚段-柔性段载荷传递参数，而不是继续调整末端边界或力矩作用点。",
            "",
            "## 下一步计划",
            "",
            "1. 对 n1 给出阶段性不可发布边界：当前模型下 position-only 也无法通过载荷传播门禁。",
            "2. 若继续模型修复，优先设计“显式接口力传递参数”，并要求该参数有物理含义，不能作为自由松弛变量替代载荷链。",
            "3. 转向两点 n2 的三维末端位置灵敏度稳健化，同时保留 n1 为模型修复案例。",
            "",
            f"- 本轮耗时：`{wall_s:.2f} s`",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    case = _recover_n1_case()
    force_rows, torque_rows, summary = _audit_interfaces(case)
    wall_s = float(time.perf_counter() - t0)
    _write_csv(OUT_DIR / "force_closure_rows.csv", force_rows)
    _write_csv(OUT_DIR / "torque_model_rows.csv", torque_rows)
    _write_csv(OUT_DIR / "summary.csv", [{**summary["gate"], "wall_s": wall_s}])
    payload = {
        "experiment": OUT_DIR.name,
        "status": "diagnostic_only_not_publishable",
        "wall_s": wall_s,
        "force_rows": force_rows,
        "torque_rows": torque_rows,
        "summary": summary,
        "main_conclusion": "端点载荷修正不能改变 n1 主导的力连续性失败；当前单点样本仍不可发布。",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "failure_class": "刚性磁体段载荷传播失配",
                    "evidence": {
                        "force_ratio": summary["max_force_jump_over_rigid_force"],
                        "threshold": 1.5,
                        "endpoint_load_can_fix_force_gate": False,
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    _plot(force_rows, torque_rows)
    (OUT_DIR / "record.md").write_text(_record(force_rows, torque_rows, summary, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
