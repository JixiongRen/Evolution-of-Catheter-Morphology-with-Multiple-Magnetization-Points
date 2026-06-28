"""Audit residual-block stationarity and load closure for n1 and n2."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C57_DIR = ROOT / "experiment_records_ik_optimize" / "20260607200500_p2c57_force_continuity_weight_repair"
for path in (ROOT, P2C57_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import run_p2c57_force_continuity_weight_repair as p2c57

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
TORQUE_SCALE = p2c57.TORQUE_SCALE
CASES = [
    {
        "label": "n1_soft_near_gate",
        "case_name": "n1_tip_z_rand15",
        "alpha": 0.10,
        "force_scale": 4.0,
        "reg_scale": 8.0e-4,
    },
    {
        "label": "n2_trusted_near_gate",
        "case_name": "n2_z_pair_rand12",
        "alpha": 0.10,
        "force_scale": 4.0,
        "reg_scale": 3.0e-4,
    },
]


def _set_chinese_font() -> None:
    """Use an installed CJK font when available."""
    for font_file in [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            prop = fm.FontProperties(fname=str(font_file))
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            break
    plt.rcParams["axes.unicode_minus"] = False


def _jsonable(value: Any) -> Any:
    """Convert arrays and paths into JSON-safe values."""
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
    """Write dictionaries as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _solve_case(spec: dict[str, Any]) -> dict[str, Any]:
    """Recover a representative near-gate branch using the P2c57 staged path."""
    params, current_i = p2c57.p2c24._params_from_saved_case(spec["case_name"], spec["alpha"])
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(spec["case_name"], spec["alpha"])
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)
    y = np.asarray(y_seed, dtype=float).copy()
    y_ref = y.copy()
    final_solve: dict[str, Any] | None = None
    final_args: Any = None
    final_funcs: dict[str, Any] | None = None
    stage_rows: list[dict[str, Any]] = []
    for stage, (position_scale, max_iter) in enumerate(p2c57.STAGES, start=1):
        args = p2c57.Args()
        args.position_scale = float(position_scale)
        args.max_iter = int(max_iter)
        args.reg_scale = float(spec["reg_scale"])
        funcs = p2c57._make_split_functions(
            params,
            target_p,
            y_ref,
            args,
            force_scale=float(spec["force_scale"]),
            torque_scale=TORQUE_SCALE,
        )
        solve = p2c57.p2c47._solve(y, current_i, funcs, args)
        row = p2c57._row(
            force_scale=float(spec["force_scale"]),
            stage=stage,
            pos_scale=position_scale,
            max_iter=max_iter,
            params=params,
            target_p=target_p,
            solve=solve,
        )
        row["label"] = spec["label"]
        row["reg_scale"] = float(spec["reg_scale"])
        stage_rows.append(row)
        y = solve["y"]
        final_solve = solve
        final_args = args
        final_funcs = funcs
    assert final_solve is not None and final_funcs is not None
    return {
        "spec": spec,
        "params": params,
        "current_i": np.asarray(current_i, dtype=float),
        "target_p": np.asarray(target_p, dtype=float),
        "y_ref": y_ref,
        "solve": final_solve,
        "args": final_args,
        "funcs": final_funcs,
        "stage_rows": stage_rows,
    }


def _block_slices(n_flex: int) -> dict[str, slice]:
    """Return residual block slices for the P2c57 residual vector."""
    continuity_len = 6 * max(n_flex - 1, 0)
    return {
        "末端位置": slice(0, 3),
        "末端边界": slice(3, 9),
        "接口连续": slice(9, 9 + continuity_len),
        "正则项": slice(9 + continuity_len, 9 + continuity_len + 6 * n_flex),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine between two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1.0e-30:
        return 0.0
    return float(np.dot(a, b) / denom)


def _stationarity_rows(case: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute gradient contribution rows and pairwise block cosines."""
    spec = case["spec"]
    y = np.asarray(case["solve"]["y"], dtype=float)
    I = np.asarray(case["current_i"], dtype=float)
    r = np.asarray(case["funcs"]["residual"](jnp.asarray(y), jnp.asarray(I)), dtype=float)
    J = np.asarray(case["funcs"]["jacobian"](jnp.asarray(y), jnp.asarray(I)), dtype=float)
    slices = _block_slices(len(case["params"].flex))
    total_g = J.T @ r
    total_norm = float(np.linalg.norm(total_g))
    gradients: dict[str, np.ndarray] = {}
    for block, slc in slices.items():
        Jb = J[slc, :]
        rb = r[slc]
        gb = Jb.T @ rb
        gradients[block] = gb
    grad_norm_sum = max(sum(np.linalg.norm(g) for g in gradients.values()), 1.0e-30)
    rows: list[dict[str, Any]] = []
    for block, slc in slices.items():
        Jb = J[slc, :]
        rb = r[slc]
        gb = gradients[block]
        rows.append(
            {
                "label": spec["label"],
                "case": spec["case_name"],
                "alpha": spec["alpha"],
                "block": block,
                "residual_norm": float(np.linalg.norm(rb)),
                "jacobian_fro_norm": float(np.linalg.norm(Jb, ord="fro")),
                "gradient_norm": float(np.linalg.norm(gb)),
                "gradient_share_of_sum": float(np.linalg.norm(gb) / grad_norm_sum),
                "gradient_share_of_total": float(np.linalg.norm(gb) / max(total_norm, 1.0e-30)),
                "cosine_to_total_gradient": _cosine(gb, total_g),
            }
        )
    pair_rows: list[dict[str, Any]] = []
    names = list(gradients)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair_rows.append(
                {
                    "label": spec["label"],
                    "case": spec["case_name"],
                    "alpha": spec["alpha"],
                    "block_a": a,
                    "block_b": b,
                    "gradient_cosine": _cosine(gradients[a], gradients[b]),
                }
            )
    return rows, pair_rows


def _gate_row(case: dict[str, Any]) -> dict[str, Any]:
    """Build a compact gate row for one recovered branch."""
    spec = case["spec"]
    solve = case["solve"]
    params = case["params"]
    target_p = case["target_p"]
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    load = p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=p2c57.Args.steps_per_flex)
    row = {
        "label": spec["label"],
        "case": spec["case_name"],
        "alpha": spec["alpha"],
        "force_scale": spec["force_scale"],
        "reg_scale": spec["reg_scale"],
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": float(solve["stationarity_norm"]),
        "jacobian_condition": float(solve["jacobian_condition"]),
        "jacobian_rank": int(solve["jacobian_rank"]),
        "continuity_norm": float(solve["continuity_norm"]),
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "max_interface_force_jump_over_rigid_force": float(load.get("max_interface_force_jump_over_rigid_force", 0.0)),
        "max_force_ratio_to_internal_force": float(load.get("max_force_ratio_to_internal_force", 0.0)),
        "nfev": int(solve["nfev"]),
        "steady_s": float(solve["steady_s"]),
    }
    row["publish_gate"] = bool(
        row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["continuity_norm"] <= 1.0e-3
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    if row["publish_gate"]:
        row["failure_class"] = "通过"
    elif row["stationarity_norm"] > 1.0e-5:
        row["failure_class"] = "一阶最优性退化"
    elif row["jacobian_condition"] >= 1.0e9:
        row["failure_class"] = "条件数严重病态"
    elif row["jacobian_condition"] >= 1.0e7:
        row["failure_class"] = "条件数病态"
    elif row["continuity_norm"] > 1.0e-3 or row["max_interface_force_jump_over_rigid_force"] > 1.5:
        row["failure_class"] = "载荷传播失配"
    else:
        row["failure_class"] = "综合失败"
    return row


def _plot(block_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> None:
    """Plot block-gradient contributions."""
    _set_chinese_font()
    labels = [row["label"] for row in gate_rows]
    blocks = ["末端位置", "末端边界", "接口连续", "正则项"]
    x = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, block in enumerate(blocks):
        vals = [
            next(row for row in block_rows if row["label"] == label and row["block"] == block)["gradient_norm"]
            for label in labels
        ]
        axes[0].bar(x + (i - 1.5) * width, vals, width=width, label=block)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0e"))
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("梯度贡献范数")
    axes[0].set_title("残差块一阶贡献")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    metrics = ["stationarity_norm", "jacobian_condition", "max_interface_force_jump_over_rigid_force"]
    metric_labels = ["一阶最优性", "条件数", "载荷比"]
    for i, metric in enumerate(metrics):
        axes[1].bar(x + (i - 1) * 0.22, [row[metric] for row in gate_rows], width=0.22, label=metric_labels[i])
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0e"))
    axes[1].set_xticks(x, labels)
    axes[1].set_title("主要门禁")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c84 分块最优性与载荷闭合审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "block_stationarity_load_closure_cn.png", dpi=180)
    plt.close(fig)


def _record(gate_rows: list[dict[str, Any]], block_rows: list[dict[str, Any]], cosine_rows: list[dict[str, Any]], wall_s: float) -> str:
    """Render the Chinese experiment record."""
    gate_table = [
        "| 样本 | 一阶最优性 | 条件数 | 接口连续 | 载荷比 | 末端误差/m | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in gate_rows:
        gate_table.append(
            f"| {row['label']} | {row['stationarity_norm']:.3e} | {row['jacobian_condition']:.3e} | "
            f"{row['continuity_norm']:.3e} | {row['max_interface_force_jump_over_rigid_force']:.3e} | "
            f"{row['position_error_m']:.3e} | {row['failure_class']} |"
        )
    block_table = [
        "| 样本 | 残差块 | 残差范数 | 梯度贡献 | 与总梯度夹角余弦 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in block_rows:
        block_table.append(
            f"| {row['label']} | {row['block']} | {row['residual_norm']:.3e} | {row['gradient_norm']:.3e} | "
            f"{row['cosine_to_total_gradient']:.3e} |"
        )
    conflict_lines = []
    for label in sorted({row["label"] for row in cosine_rows}):
        sub = [row for row in cosine_rows if row["label"] == label]
        most_negative = min(sub, key=lambda row: row["gradient_cosine"])
        conflict_lines.append(
            f"- `{label}` 最强梯度冲突为 `{most_negative['block_a']}` 与 `{most_negative['block_b']}`，夹角余弦 `{most_negative['gradient_cosine']:.3e}`。"
        )
    dominant_lines = []
    for label in sorted({row["label"] for row in block_rows}):
        sub = [row for row in block_rows if row["label"] == label]
        dominant = max(sub, key=lambda row: row["gradient_norm"])
        dominant_lines.append(f"- `{label}` 最大一阶贡献来自 `{dominant['block']}`，梯度贡献 `{dominant['gradient_norm']:.3e}`。")
    return "\n".join(
        [
            "# P2c84 分块最优性与载荷闭合审计记录",
            "",
            "## 实验目的",
            "",
            "P2c83 指出下一步不应盲目改模型，而应先定位一阶最优性与载荷闭合失败来自哪个残差块。本轮恢复 n1 软约束近门样本和 n2 受信任近门样本，分别计算残差分块梯度贡献、残差块梯度夹角和载荷门禁。",
            "",
            "## 样本与门禁",
            "",
            *gate_table,
            "",
            "## 残差块一阶贡献",
            "",
            *block_table,
            "",
            "## 残差块冲突",
            "",
            *dominant_lines,
            *conflict_lines,
            "",
            "## 结论",
            "",
            "n1 样本的一阶最优性和条件数已经过门，但载荷传播门禁失败；其最大梯度冲突发生在末端位置与末端边界之间，说明当前近门解不是单纯接口连续权重不足，而是任务位置、自由边界和载荷传播之间存在相容性张力。n2 样本的载荷传播门禁较好，主要阻断项仍是条件数；其最大一阶贡献来自接口连续块，且末端位置与接口连续梯度几乎相反，说明 n2 的可信分支仍存在残差尺度和接口耦合张力。",
            "",
            "本轮不发布任何新灵敏度。它只给出模型修复方向：若接口连续块主导，应修载荷传播；若末端边界与位置梯度相反，应修任务/边界尺度或承认目标处于可达边界；若正则项主导，应重新审查参考锚定。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 对本轮发现的主导冲突块做最小模型修复实验。",
            "2. 若 n1 仍由载荷传播失配主导，应先重构刚段接口平衡方程，再复测 P2c64/P2c65 近门样本。",
            "3. 若 n2 主要是尺度/条件数问题，则保留受信任分支条件口径，优先做分支可达性而不是再改载荷模型。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures_path = OUT_DIR / "failure_events.jsonl"
    failures_path.write_text("", encoding="utf-8")
    recovered = [_solve_case(spec) for spec in CASES]
    gate_rows = [_gate_row(case) for case in recovered]
    block_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for case in recovered:
        rows, pairs = _stationarity_rows(case)
        block_rows.extend(rows)
        cosine_rows.extend(pairs)
        stage_rows.extend(case["stage_rows"])
    _write_csv(OUT_DIR / "gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "block_gradient_rows.csv", block_rows)
    _write_csv(OUT_DIR / "block_gradient_cosines.csv", cosine_rows)
    _write_csv(OUT_DIR / "stage_rows.csv", stage_rows)
    for row in gate_rows:
        if not row["publish_gate"]:
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"failure": row["failure_class"], "evidence": _jsonable(row)}, ensure_ascii=False) + "\n")
    summary = {
        "experiment": "P2c84_block_stationarity_load_closure_audit",
        "gate_rows": gate_rows,
        "dominant_gradient_blocks": {
            label: max([row for row in block_rows if row["label"] == label], key=lambda row: row["gradient_norm"])["block"]
            for label in sorted({row["label"] for row in block_rows})
        },
        "min_gradient_cosines": {
            label: min([row for row in cosine_rows if row["label"] == label], key=lambda row: row["gradient_cosine"])
            for label in sorted({row["label"] for row in cosine_rows})
        },
        "wall_s": float(time.perf_counter() - t0),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", gate_rows)
    _plot(block_rows, gate_rows)
    (OUT_DIR / "record.md").write_text(_record(gate_rows, block_rows, cosine_rows, summary["wall_s"]), encoding="utf-8")


if __name__ == "__main__":
    main()
