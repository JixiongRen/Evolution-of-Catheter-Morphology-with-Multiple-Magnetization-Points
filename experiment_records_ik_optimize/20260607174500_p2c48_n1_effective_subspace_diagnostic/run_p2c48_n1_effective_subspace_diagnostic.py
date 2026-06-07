"""Diagnose n1 effective position-response subspace."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C47_DIR = ROOT / "experiment_records_ik_optimize" / "20260607172000_p2c47_explicit_continuity_position_gate"
for path in (ROOT, FK_OPT_DIR, P2C47_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents

import run_p2c47_explicit_continuity_position_gate as p2c47

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
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


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _effective_rank(svals: np.ndarray, *, rel_tol: float = 0.1) -> int:
    if svals.size == 0 or float(svals[0]) <= 0.0:
        return 0
    return int(np.sum(svals >= rel_tol * float(svals[0])))


def _response_metrics(matrix: np.ndarray) -> dict[str, Any]:
    u, svals, vt = np.linalg.svd(matrix, full_matrices=False)
    cond = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf")
    rank_10 = _effective_rank(svals, rel_tol=0.1)
    rank_01 = _effective_rank(svals, rel_tol=0.01)
    energy = svals**2
    total = max(float(np.sum(energy)), 1.0e-30)
    return {
        "singular_values": svals.tolist(),
        "condition": cond,
        "effective_rank_rel_0p1": rank_10,
        "effective_rank_rel_0p01": rank_01,
        "energy_fraction": (energy / total).tolist(),
        "left_singular_vectors": u.tolist(),
        "right_singular_vectors": vt.tolist(),
    }


def _make_sensitivity_functions(params_base: Any, target_p: np.ndarray, y_ref: np.ndarray, args: Any) -> dict[str, Any]:
    funcs = p2c47._make_functions(
        params_base,
        target_p,
        y_ref,
        args,
        continuity_scale=1.0,
        boundary_mode="full",
    )

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def task_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = p2c47.forward_segment_wrench(params_for_I(I), y, steps_per_flex=args.steps_per_flex)
        return out.tip_p.reshape(3)

    return {
        **funcs,
        "task": jax.jit(task_yI),
        "task_y": jax.jit(jax.jacfwd(task_yI, argnums=0)),
        "task_I": jax.jit(jax.jacfwd(task_yI, argnums=1)),
        "residual_I": jax.jit(jax.jacfwd(funcs["residual"]._fun, argnums=1)) if hasattr(funcs["residual"], "_fun") else None,
    }


def main() -> None:
    t0 = time.perf_counter()

    class Args:
        boundary_scale = 0.25
        reg_scale = 1.0e-4
        max_iter = 120
        steps_per_flex = 4
        position_scale = 60.0

    args = Args()
    case_name = "n1_tip_z_rand15"
    alpha = 0.10
    params, I0 = p2c47.p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_slack = p2c47.p2c24._load_saved_target_and_seed(case_name, alpha)
    y0 = p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=args.steps_per_flex)
    funcs = p2c47._make_functions(params, target_p, y0, args, continuity_scale=1.0, boundary_mode="full")
    solve = p2c47._solve(y0, I0, funcs, args)

    y = np.asarray(solve["y"], dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def task_yI(y_in: jnp.ndarray, I_in: jnp.ndarray) -> jnp.ndarray:
        out = p2c47.forward_segment_wrench(params_for_I(I_in), y_in, steps_per_flex=args.steps_per_flex)
        return out.tip_p.reshape(3)

    residual_jit = funcs["residual"]
    Jy = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    JI = np.asarray(jax.device_get(jax.jacfwd(lambda yy, II: residual_jit(yy, II), argnums=1)(y_j, I_j)), dtype=float)
    Ty = np.asarray(jax.device_get(jax.jacfwd(task_yI, argnums=0)(y_j, I_j)), dtype=float)
    TI = np.asarray(jax.device_get(jax.jacfwd(task_yI, argnums=1)(y_j, I_j)), dtype=float)
    direct = TI

    hessian = Jy.T @ Jy
    rhs = Jy.T @ JI
    ridge = 1.0e-8 * max(float(np.linalg.norm(hessian, ord=2)), 1.0)
    dy_dI = -np.linalg.solve(hessian + ridge * np.eye(hessian.shape[0]), rhs)
    implicit = Ty @ dy_dI + TI

    baseline = {
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "continuity_norm": solve["continuity_norm"],
        "terminal_boundary_norm": float(np.linalg.norm(np.asarray(solve["out"]["boundary"], dtype=float))),
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "baseline_publishable": False,
        "baseline_blocker": "一阶最优性、条件数和接口载荷传播门禁未通过",
    }
    direct_metrics = _response_metrics(direct)
    implicit_metrics = _response_metrics(implicit)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, direct_metrics["singular_values"], width=width, label="固定状态响应")
    ax.bar(x + width / 2, implicit_metrics["singular_values"], width=width, label="近似隐式响应")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["第1方向", "第2方向", "第3方向"])
    ax.set_ylabel("奇异值")
    ax.set_title("单点工况末端位置响应有效子空间")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_effective_subspace_spectrum_cn.png", dpi=180)
    plt.close(fig)

    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "case": case_name,
        "alpha": alpha,
        "baseline": baseline,
        "direct_position_response": direct_metrics,
        "implicit_position_response": implicit_metrics,
        "interpretation": {
            "publishability": "不可发布",
            "reason": "基线物理门禁未通过；有效子空间结果只能作为诊断证据",
            "single_magnet_effective_rank_direct_rel_0p1": direct_metrics["effective_rank_rel_0p1"],
            "single_magnet_effective_rank_implicit_rel_0p1": implicit_metrics["effective_rank_rel_0p1"],
        },
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "jax_devices": [str(device) for device in jax.devices()],
    }
    _write_json(OUT_DIR / "summary.json", payload)
    (OUT_DIR / "summary.csv").write_text(
        "case,alpha,baseline_publishable,direct_rank_0p1,implicit_rank_0p1,direct_condition,implicit_condition,stationarity_norm,jacobian_condition\n"
        f"{case_name},{alpha},False,{direct_metrics['effective_rank_rel_0p1']},{implicit_metrics['effective_rank_rel_0p1']},"
        f"{direct_metrics['condition']},{implicit_metrics['condition']},{baseline['stationarity_norm']},{baseline['jacobian_condition']}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "failure_events.jsonl").write_text(
        json.dumps(
            {
                "ts": time.time(),
                "failure": "有效子空间仅可诊断",
                "evidence": baseline,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "record.md").write_text(_record(payload), encoding="utf-8")
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


def _record(payload: dict[str, Any]) -> str:
    direct = payload["direct_position_response"]
    implicit = payload["implicit_position_response"]
    baseline = payload["baseline"]
    return "\n".join(
        [
            "# P2c48 单点有效子空间诊断记录",
            "",
            "## 实验目的",
            "",
            "在 P2c47 未获得可发布基线后，本轮针对单点磁控工况计算末端位置响应的有效子空间，判断其是否更适合降维灵敏度路线。本轮结果仅作为诊断，不能作为发布级灵敏度。",
            "",
            "## 实验方案",
            "",
            "复用 P2c47 的显式接口连续模型和最优组合：`n1_tip_z_rand15`、`alpha=0.10`、力矩全闭合、接口连续权重 `1`。在该解附近计算两类末端位置响应矩阵：固定状态响应和近似隐式响应。固定状态响应表示不重新平衡时电流对末端位置的直接影响；近似隐式响应表示用高斯牛顿线性化估计重新平衡后的局部影响。",
            "",
            "## 结果",
            "",
            f"- 基线是否可发布：`{baseline['baseline_publishable']}`",
            f"- 基线阻断原因：{baseline['baseline_blocker']}",
            f"- 一阶最优性残差：`{baseline['stationarity_norm']:.3e}`",
            f"- 条件数：`{baseline['jacobian_condition']:.3e}`",
            f"- 接口连续残差：`{baseline['continuity_norm']:.3e}`",
            f"- 固定状态响应奇异值：`{[round(x, 6) for x in direct['singular_values']]}`",
            f"- 固定状态 10% 相对阈值有效秩：`{direct['effective_rank_rel_0p1']}`",
            f"- 近似隐式响应奇异值：`{[round(x, 6) for x in implicit['singular_values']]}`",
            f"- 近似隐式 10% 相对阈值有效秩：`{implicit['effective_rank_rel_0p1']}`",
            "",
            "## 结论",
            "",
            "单点工况在当前基线下仍不可发布，因为物理门禁没有通过。因此，本轮有效子空间结果只能说明局部响应结构，不能证明可发布灵敏度。若固定状态和近似隐式响应均显示有效秩小于 3，则后续不应强推完整三维末端位置灵敏度；若有效秩接近 3，也仍需先修复基线物理门禁。",
            "",
            "## 下一步计划",
            "",
            "1. 对单点工况停止完整三维灵敏度发布尝试，转为有效子空间或不可发布边界论证。",
            "2. 对两点工况优先做显式连续延拓修复，因为 P2c47 中两点已有一阶最优性改善迹象。",
            "",
        ]
    )


if __name__ == "__main__":
    main()
