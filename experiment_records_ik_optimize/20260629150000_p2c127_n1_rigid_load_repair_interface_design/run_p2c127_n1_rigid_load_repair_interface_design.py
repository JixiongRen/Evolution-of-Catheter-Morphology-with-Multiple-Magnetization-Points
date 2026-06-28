"""Generate the n1 rigid-load repair interface design package."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write stable JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_code_map() -> list[dict[str, Any]]:
    """Build the code-level map for the proposed repair."""
    return [
        {
            "component": "FK 刚段传播参考",
            "file": "pose_modules_nondim/equilibrium_solver_nondim.py",
            "symbol": "EquilibriumSolverNondim.residual",
            "current_role": "对每个刚段计算自动外载，并调用 rigid.propagate 得到刚段远端状态。",
            "repair_use": "作为 IK 显式刚段载荷传播方程的物理参考。",
        },
        {
            "component": "IK 位置与连续残差",
            "file": "experiment_records_ik_optimize/20260607200500_p2c57_force_continuity_weight_repair/run_p2c57_force_continuity_weight_repair.py",
            "symbol": "_make_split_functions",
            "current_role": "残差由末端位置、边界、interface_wrench_residual 和正则组成。",
            "repair_use": "新增刚段载荷传播残差，不再只靠 force_scale 或 beta 比例吸收载荷链缺口。",
        },
        {
            "component": "硬力连续反事实",
            "file": "experiment_records_ik_optimize/20260629030000_p2c110_n1_hard_force_scaled_continuity/run_p2c110_n1_hard_force_scaled_continuity.py",
            "symbol": "FORCE_SCALES",
            "current_role": "扫描 1、4、16 倍力连续权重。",
            "repair_use": "作为修复模型的负对照，证明调权重不能替代物理传播。",
        },
        {
            "component": "beta 载荷传递反事实",
            "file": "experiment_records_ik_optimize/20260629033000_p2c111_n1_beta_load_transfer/summary.json",
            "symbol": "beta_reg_scales",
            "current_role": "用自由比例参数近似载荷传递。",
            "repair_use": "作为自由松弛变量不可发布的负对照。",
        },
    ]


def build_equation_rows() -> list[dict[str, Any]]:
    """Build proposed residual equations."""
    return [
        {
            "equation": "刚段平衡传播",
            "residual": "r_x = x_rigid_end - Phi_rigid(x_flex_end, F_m, tau_m)",
            "meaning": "刚段远端状态必须由柔性段末端状态和磁体段合外载传播得到。",
            "gate": "几何残差进入基线残差；不允许只由正则项吸收。",
        },
        {
            "equation": "力连续",
            "residual": "r_F = n_flex_distal + F_m",
            "meaning": "柔性段远端内力应平衡刚性磁体段对接口的等效外力。",
            "gate": "||r_F|| / max(||F_m||, floor) <= 1.5。",
        },
        {
            "equation": "矩连续",
            "residual": "r_M = m_flex_distal + tau_m + r_cm x F_m",
            "meaning": "柔性段远端内矩应平衡外力矩和力臂引起的附加力矩。",
            "gate": "与力连续同时满足；不能只满足矩而破坏力。",
        },
        {
            "equation": "任务残差",
            "residual": "r_p = p_tip - p_target",
            "meaning": "末端位置拟合仍保留，但不能覆盖物理闭合失败。",
            "gate": "任务维度不得超过有效秩；基线物理门禁不过时不扫描扰动。",
        },
    ]


def build_milestone_rows() -> list[dict[str, Any]]:
    """Build a staged implementation plan."""
    return [
        {
            "stage": "P2c128",
            "goal": "实现只读载荷传播观测器",
            "work": "在不改求解残差的情况下输出 F_m、tau_m、r_F、r_M、力臂项和现有 interface_wrench_residual 的差异。",
            "success": "复现实验 P2c99/P2c120 的主失败，并定位力连续与矩连续各自贡献。",
        },
        {
            "stage": "P2c129",
            "goal": "显式传播残差最小原型",
            "work": "在实验级 _make_split_functions 分支中加入 r_F/r_M/r_x，不替换生产 IK。",
            "success": "n1 基线同时接近 stationarity <=1e-5、condition <1e7、force ratio <=1.5。",
        },
        {
            "stage": "P2c130",
            "goal": "任务降维对照",
            "work": "若三维位置仍失败，测试一维或二维有效子空间任务，不跑完整位姿。",
            "success": "有效秩与任务维度一致，且扰动重求解误差 <0.05。",
        },
        {
            "stage": "P2c131",
            "goal": "负样本复核",
            "work": "若连续三轮仍同类失败，固定当前 n1 为不可发布边界，不再扫参。",
            "success": "输出模型修复失败证明和工作空间建议。",
        },
    ]


def plot_flow() -> None:
    """Plot the repair flow diagram."""
    configure_matplotlib_cjk()
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    ax.axis("off")
    boxes = [
        (0.05, 0.65, "现状\n权重/比例修补\n不可发布"),
        (0.30, 0.65, "P2c128\n只读载荷传播观测器"),
        (0.55, 0.65, "P2c129\n显式刚段传播残差"),
        (0.80, 0.65, "基线门禁\n物理闭合+条件数"),
        (0.55, 0.25, "P2c130\n任务降维对照"),
        (0.80, 0.25, "扰动灵敏度\n仅基线通过后"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f2efe8", "edgecolor": "#3b3b3b"},
            fontsize=11,
        )
    arrows = [
        ((0.15, 0.65), (0.24, 0.65)),
        ((0.40, 0.65), (0.49, 0.65)),
        ((0.65, 0.65), (0.74, 0.65)),
        ((0.80, 0.57), (0.80, 0.33)),
        ((0.65, 0.25), (0.74, 0.25)),
        ((0.55, 0.57), (0.55, 0.33)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_title("n1 显式刚性磁体段载荷传播修复路线", fontsize=14)
    fig.savefig(OUT_DIR / "n1_rigid_load_repair_flow_cn.png", dpi=220)
    plt.close(fig)


def write_record(code_rows: list[dict[str, Any]], equation_rows: list[dict[str, Any]], milestone_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write the Chinese design record."""
    code_lines = "\n".join(
        f"- {row['component']}：`{row['file']}` / `{row['symbol']}`。当前作用：{row['current_role']} 修复用途：{row['repair_use']}"
        for row in code_rows
    )
    equation_lines = "\n".join(
        f"- {row['equation']}：{row['residual']}。含义：{row['meaning']} 验收：{row['gate']}"
        for row in equation_rows
    )
    milestone_lines = "\n".join(
        f"- {row['stage']}：{row['goal']}。工作：{row['work']} 验收：{row['success']}"
        for row in milestone_rows
    )
    record = f"""# P2c127 n1 显式刚性磁体段载荷传播修复接口设计

## 实验目的

P2c126 已判定单点磁控当前模型不可发布，主失败为刚性磁体段载荷传播失配与模型耦合缺失。本轮不直接修改求解器，而是定位代码接口并设计最小修复路线，保证下一轮实现只改变物理方程，不继续依赖权重调参或自由比例变量。

## 代码定位

{code_lines}

## 方程设计

当前实验层残差可以写为：

\\[
r = [r_p, r_b, w_F r_c^F, w_M r_c^M, r_{{reg}}],
\\]

其中 \(w_F\) 和 \(w_M\) 是权重。历史实验说明，仅改变权重会把失配转移到一阶最优性、条件数或接口残差。修复版应显式引入刚性磁体段载荷传播：

{equation_lines}

## 实施阶段

{milestone_lines}

## 不变量

- 不在基线门禁前做扰动灵敏度扫描。
- 不用自由比例变量替代真实力链传播。
- 不扩大到完整末端位姿任务。
- 不替换生产 IK；先在实验级 runner 中验证。
- 每个候选必须同时记录一阶最优性、条件数、载荷传播、接口残差、中心线分支漂移和局部线性化误差。

## 当前结论

本轮给出的是模型修复接口设计，不声明 n1 已可发布。下一轮应先实现只读载荷传播观测器，验证刚段外载、力臂矩和接口内力链之间的数值差异；只有该观测器复现 P2c126 失败机制后，才进入显式传播残差最小原型。

## 下一步计划

执行 P2c128：实现只读载荷传播观测器。输出 \(F_m\)、\(\tau_m\)、\(r_F\)、\(r_M\)、力臂项、现有接口残差分块，并与 P2c120 的不可发布结论对齐。

## 运行信息

- 状态：{summary['status']}。
- 墙钟耗时：{summary['wall_s']:.4f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Generate the repair interface design package."""
    t0 = time.perf_counter()
    code_rows = build_code_map()
    equation_rows = build_equation_rows()
    milestone_rows = build_milestone_rows()
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "n1_rigid_load_repair_interface_designed",
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": False,
        "next_stage": "P2c128 readonly rigid-load propagation observer",
        "code_components": len(code_rows),
        "equations": len(equation_rows),
        "milestones": len(milestone_rows),
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "code_map_rows.csv", code_rows)
    write_csv(OUT_DIR / "equation_rows.csv", equation_rows)
    write_csv(OUT_DIR / "milestone_rows.csv", milestone_rows)
    write_csv(OUT_DIR / "summary.csv", [{"status": summary["status"], "wall_s": summary["wall_s"]}])
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    plot_flow()
    write_record(code_rows, equation_rows, milestone_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
