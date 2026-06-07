from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

P2C28_CASE = ROOT / "experiment_records_ik_optimize/20260606173500_p2c28_cross_condition_reproducibility_map/case_summary.csv"
P2C10N_BASE = ROOT / "experiment_records_ik_optimize/20260605040948_p2c10n_position_medium_current_scan/base_summary.csv"
P2C22_SUMMARY = ROOT / "experiment_records_ik_optimize/20260606152302_p2c22_cross_condition_reliability_pilot/summary.csv"

CASES = {"n1_tip_z_rand15", "n2_z_pair_rand12"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def to_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def git_info() -> dict[str, str]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def classify_p2c28(row: dict[str, str]) -> dict[str, object]:
    case = row["case"]
    magnet_count = int(row["magnet_count"])
    stationarity = to_float(row["baseline_stationarity_norm"])
    slack_ratio = to_float(row["baseline_slack_force_ratio"])
    condition = to_float(row["baseline_jacobian_condition"])
    branch = to_float(row["branch_rms_max"])
    linearity = to_float(row["max_linearization_error"])
    publishable = to_bool(row["publishable_gate"])

    failures: list[str] = []
    if stationarity > 1e-5:
        failures.append("一阶最优性退化")
    if condition > 1e9:
        failures.append("条件数病态")
    elif condition > 1e7:
        failures.append("条件数偏高")
    if slack_ratio > 2.0:
        failures.append("刚性磁体段载荷传播失配")
    if branch > 1e-3:
        failures.append("分支漂移")
    if linearity >= 0.05:
        failures.append("局部线性化失效")
    if not publishable and not failures:
        failures.append("证据不足")

    return {
        "source": "P2c28",
        "case": case,
        "magnet_count": magnet_count,
        "task": "position-only sensitivity replay",
        "alpha": to_float(row["alpha"]),
        "eps_alpha": 0.01,
        "publishable": publishable,
        "stationarity_norm": stationarity,
        "condition": condition,
        "slack_force_ratio": slack_ratio,
        "branch_rms_m": branch,
        "linearization_error": linearity,
        "sensitivity_norm": to_float(row["sensitivity_norm_mean"]),
        "primary_failure": row.get("primary_failure_mode") or (failures[0] if failures else ""),
        "failure_modes": ";".join(failures),
        "recommendation": recommendation(case, magnet_count, failures),
    }


def classify_p2c10n(row: dict[str, str]) -> dict[str, object]:
    case = row["case"]
    magnet_count = int(row["active_magnet_count"])
    condition = to_float(row["hessian_condition"])
    stationarity = to_float(row["base_stationarity_norm"])
    rank = int(row["rank"])
    rank_ok = to_bool(row["rank_ok"])
    required_rank = int(row["required_rank"])
    base_success = to_bool(row["base_success"])

    failures: list[str] = []
    if not base_success:
        failures.append("基线求解未通过")
    if stationarity > 1e-5:
        failures.append("一阶最优性退化")
    if condition > 1e9:
        failures.append("条件数病态")
    elif condition > 1e7:
        failures.append("条件数偏高")
    if not rank_ok or rank < required_rank:
        failures.append("有效秩不足")

    return {
        "source": "P2c10n",
        "case": case,
        "magnet_count": magnet_count,
        "task": "position-only rank and baseline audit",
        "alpha": to_float(row["alpha"]),
        "current_max_abs": to_float(row["current_max_abs"]),
        "publishable": False,
        "base_success": base_success,
        "stationarity_norm": stationarity,
        "condition": condition,
        "effective_rank": rank,
        "required_rank": required_rank,
        "rank_ok": rank_ok,
        "sensitivity_norm": to_float(row["sensitivity_fro_norm"]),
        "primary_failure": row["base_failure_class"],
        "failure_modes": ";".join(failures),
        "recommendation": recommendation(case, magnet_count, failures),
    }


def recommendation(case: str, magnet_count: int, failures: list[str]) -> str:
    if "刚性磁体段载荷传播失配" in failures:
        return "停止扰动扫描，优先做接口物理闭合或载荷传播重参数化。"
    if "有效秩不足" in failures and magnet_count == 1:
        return "仅发布有效子空间灵敏度；不得宣称完整三维位置灵敏度。"
    if "条件数病态" in failures:
        return "先做任务降维和病态方向识别，不做大规模扫描。"
    if "一阶最优性退化" in failures:
        return "先做基线精修和残差分块归因，再进入扰动重求解。"
    if not failures:
        return "可进入小扰动复核，优先 eps=0.005 和 0.010。"
    return "保留为诊断样本，继续最小归因闭环。"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_events(rows: list[dict[str, object]]) -> None:
    with (OUT_DIR / "failure_events.jsonl").open("w") as f:
        for row in rows:
            if row.get("publishable") is True:
                continue
            event = {
                "case": row["case"],
                "source": row["source"],
                "failure_class": row.get("primary_failure") or row.get("failure_modes"),
                "failure_modes": row.get("failure_modes", ""),
                "recommendation": row.get("recommendation", ""),
            }
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def plot_rows(rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter

    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False

    p2c28 = [row for row in rows if row["source"] == "P2c28"]
    labels = [f"{row['case']}\n磁体数={row['magnet_count']}" for row in p2c28]
    stationarity = [float(row["stationarity_norm"]) for row in p2c28]
    slack = [float(row["slack_force_ratio"]) for row in p2c28]
    condition = [float(row["condition"]) for row in p2c28]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=160)

    def sci_tick(value: float, _position: int) -> str:
        if value <= 0:
            return ""
        exponent = int(round(math.log10(value)))
        return f"1e{exponent}"

    axes[0].bar(labels, stationarity, color="#8c564b")
    axes[0].axhline(1e-5, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(sci_tick))
    axes[0].set_title("一阶最优性")
    axes[0].set_ylabel("残差范数")

    axes[1].bar(labels, slack, color="#d62728")
    axes[1].axhline(2.0, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(1.5, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title("载荷传播比例")

    axes[2].bar(labels, condition, color="#1f77b4")
    axes[2].axhline(1e7, color="gray", linestyle=":", linewidth=1)
    axes[2].axhline(1e9, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_formatter(FuncFormatter(sci_tick))
    axes[2].set_title("局部条件数")

    fig.suptitle("P2c39 单点/两点末端位置灵敏度基线门禁")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "single_double_baseline_gate_cn.png")


def write_record(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    p2c28 = [row for row in rows if row["source"] == "P2c28"]
    lines = [
        "# P2c39 单点/两点末端位置灵敏度基线门禁审计",
        "",
        "## 实验目的",
        "",
        "本轮不重跑大规模 IK 求解，而是汇总已提交实验中单点、两点磁控样本的严格门禁证据，作为后续物理闭合修复和位置灵敏度实验的基线。",
        "",
        "## 数据来源",
        "",
        "- P2c28：跨工况扰动重求解可靠性图谱。",
        "- P2c10n：中等电流位置任务有效秩和条件数扫描。",
        "- P2c22：早期跨工况可靠性试探，作为历史对照。",
        "",
        "## 关键结果",
        "",
        "| 样本 | 磁体数 | 可发布 | 一阶最优性 | 载荷传播比例 | 条件数 | 线性化误差 | 主失败 |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in p2c28:
        lines.append(
            f"| {row['case']} | {row['magnet_count']} | {row['publishable']} | "
            f"{float(row['stationarity_norm']):.3e} | {float(row['slack_force_ratio']):.3f} | "
            f"{float(row['condition']):.3e} | {float(row['linearization_error']):.3e} | "
            f"{row['primary_failure']} |"
        )
    lines.extend(
        [
            "",
            "## 数据分析",
            "",
            "单点和两点样本在末端位置扰动复核中均未达到可发布门禁。单点样本主要表现为刚性磁体段载荷传播比例超过门限，同时条件数偏高；两点样本同时出现载荷传播失配、一阶最优性退化和较高条件数。",
            "",
            "这说明当前单点、两点问题不能继续只依赖末端拟合或优化器调参推进。下一阶段应优先处理接口物理闭合和有效任务维度，而不是扩大扰动扫描。",
            "",
            "## 实验结论",
            "",
            "1. 当前证据不支持发布单点或两点磁控完整末端位置灵敏度。",
            "2. 单点磁控应转为有效子空间灵敏度研究，先证明可控方向和不可控方向。",
            "3. 两点磁控应优先修复载荷传播和一阶最优性，再重测三维末端位置灵敏度。",
            "4. 基线不过门前，不应进入大规模 epsilon 扰动扫描。",
            "",
            "## 下一步计划",
            "",
            "1. 为单点、两点分别实现接口物理闭合原型，对比硬连续与载荷定标松弛。",
            "2. 固定 position-only 任务，重测基线门禁。",
            "3. 只有基线通过后，才进入 eps=0.005 和 eps=0.010 扰动重求解。",
            "4. 若连续三轮仍卡在载荷传播失配，则将对应工况标为当前模型下不可发布，并输出工程边界建议。",
            "",
            "## 产物",
            "",
            "- `single_double_baseline_gate.csv`",
            "- `single_double_rank_audit.csv`",
            "- `failure_events.jsonl`",
            "- `single_double_baseline_gate_cn.png`",
            "- `summary.json`",
        ]
    )
    (OUT_DIR / "record.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    p2c28_rows = [classify_p2c28(row) for row in read_csv(P2C28_CASE) if row["case"] in CASES]
    p2c10n_rows = [classify_p2c10n(row) for row in read_csv(P2C10N_BASE) if row["case"] in CASES]
    p2c22_rows = [row for row in read_csv(P2C22_SUMMARY) if row["case"] in CASES]

    write_csv(OUT_DIR / "single_double_baseline_gate.csv", p2c28_rows)
    write_csv(OUT_DIR / "single_double_rank_audit.csv", p2c10n_rows)
    write_csv(OUT_DIR / "p2c22_historical_baseline.csv", p2c22_rows)
    write_failure_events(p2c28_rows + p2c10n_rows)
    plot_rows(p2c28_rows + p2c10n_rows)

    summary = {
        "experiment": OUT_DIR.name,
        "wall_s": time.perf_counter() - start,
        "git": git_info(),
        "overall": {
            "p2c28_case_count": len(p2c28_rows),
            "p2c28_publishable_count": sum(1 for row in p2c28_rows if row["publishable"]),
            "p2c10n_row_count": len(p2c10n_rows),
            "cases": sorted(CASES),
        },
        "p2c28_rows": p2c28_rows,
        "p2c10n_rows": p2c10n_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(p2c28_rows + p2c10n_rows, summary)


if __name__ == "__main__":
    main()
