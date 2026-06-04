"""Recover the unstable n3_mixed_rand20 shooting baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C7_DIR = ROOT / "experiment_records_ik_optimize" / "20260604110544_p2c7_shooting_sensitivity_audit"
P2C8_DIR = ROOT / "experiment_records_ik_optimize" / "20260604114419_p2c8_continuity_branch_sensitivity"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C7_DIR, P2C8_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8

jax.config.update("jax_enable_x64", True)

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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _case_by_name(name: str) -> Any:
    for case in p2c4._cases():
        if case.name == name:
            return case
    raise ValueError(f"Unknown case: {name}")


def _target_bundle(case_name: str, *, continuity_mu: float, reg_mu: float, y_ref: np.ndarray | None = None) -> dict[str, Any]:
    case = _case_by_name(case_name)
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I0 = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    ref = y_oracle if y_ref is None else np.asarray(y_ref, dtype=float).reshape(-1)
    residual_jit, _task_jit, output_jit, _continuity_jit, Jy_jit, _JI_jit, _Ty_jit, _TI_jit = p2c8._make_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=ref,
        reg_mu=float(reg_mu),
        continuity_mu=float(continuity_mu),
        steps_per_flex=4,
    )
    _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
    return {
        "I0": I0,
        "y_oracle": y_oracle,
        "residual_jit": residual_jit,
        "output_jit": output_jit,
        "Jy_jit": Jy_jit,
    }


def _neighbor_init(target_case: str, neighbor_case: str, max_iter: int) -> np.ndarray | None:
    neighbor = _target_bundle(neighbor_case, continuity_mu=1.0, reg_mu=1.0e-4)
    solve = p2c7._solve_branch(
        y0=neighbor["y_oracle"],
        I=neighbor["I0"],
        residual_jit=neighbor["residual_jit"],
        output_jit=neighbor["output_jit"],
        Jy_jit=neighbor["Jy_jit"],
        max_iter=max_iter,
    )
    return solve.y if solve.success and solve.y.shape == neighbor["y_oracle"].shape else None


def _score(row: dict[str, Any]) -> float:
    return (
        float(row["residual_norm"])
        + 0.02 * float(row["boundary_residual_norm"])
        + 1.0e-4 * min(float(row["stationarity_norm"]), 1.0e4)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="n3_mixed_rand20")
    parser.add_argument("--neighbor", default="n3_mixed_base_i4")
    parser.add_argument("--scan_iter", type=int, default=32)
    parser.add_argument("--refine_iter", type=int, default=96)
    parser.add_argument("--top_k", type=int, default=4)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    base_for_init = _target_bundle(args.case, continuity_mu=1.0, reg_mu=1.0e-4)
    y_oracle = base_for_init["y_oracle"]
    bundle_cache: dict[tuple[float, float], dict[str, Any]] = {(1.0, 1.0e-4): base_for_init}

    def get_bundle(continuity_mu: float, reg_mu: float) -> dict[str, Any]:
        key = (float(continuity_mu), float(reg_mu))
        if key not in bundle_cache:
            bundle_cache[key] = _target_bundle(args.case, continuity_mu=key[0], reg_mu=key[1], y_ref=y_oracle)
        return bundle_cache[key]

    init_map: dict[str, np.ndarray] = {
        "oracle": y_oracle,
        "zero": np.zeros_like(y_oracle),
        "half_oracle": 0.5 * y_oracle,
    }
    neighbor_y = _neighbor_init(args.case, args.neighbor, max_iter=args.scan_iter)
    if neighbor_y is not None:
        init_map["neighbor"] = neighbor_y

    scan_specs = []
    for continuity_mu in (0.3, 1.0, 3.0, 10.0):
        for reg_mu in (0.0, 1.0e-6, 1.0e-4):
            for init_name in init_map:
                scan_specs.append((continuity_mu, reg_mu, init_name, int(args.scan_iter), "scan"))

    for continuity_mu, reg_mu, init_name, max_iter, phase in scan_specs:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": phase, "continuity_mu": continuity_mu, "reg_mu": reg_mu, "init": init_name})
        bundle = get_bundle(continuity_mu, reg_mu)
        solve = p2c7._solve_branch(
            y0=init_map[init_name],
            I=bundle["I0"],
            residual_jit=bundle["residual_jit"],
            output_jit=bundle["output_jit"],
            Jy_jit=bundle["Jy_jit"],
            max_iter=max_iter,
        )
        row = {
            "phase": phase,
            "case": args.case,
            "continuity_mu": continuity_mu,
            "reg_mu": reg_mu,
            "init": init_name,
            "max_iter": max_iter,
            "success": solve.success,
            "steady_s": solve.steady_s,
            "residual_norm": solve.residual_norm,
            "stationarity_norm": solve.stationarity_norm,
            "boundary_residual_norm": solve.boundary_norm,
            "score": 0.0,
        }
        row["score"] = _score(row)
        rows.append(row)
        if solve.success:
            break

    best_scan = sorted(rows, key=_score)[: int(args.top_k)]
    for item in best_scan:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "refine", "source": item})
        bundle = get_bundle(float(item["continuity_mu"]), float(item["reg_mu"]))
        solve = p2c7._solve_branch(
            y0=init_map[str(item["init"])],
            I=bundle["I0"],
            residual_jit=bundle["residual_jit"],
            output_jit=bundle["output_jit"],
            Jy_jit=bundle["Jy_jit"],
            max_iter=int(args.refine_iter),
        )
        row = {
            "phase": "refine",
            "case": args.case,
            "continuity_mu": item["continuity_mu"],
            "reg_mu": item["reg_mu"],
            "init": item["init"],
            "max_iter": int(args.refine_iter),
            "success": solve.success,
            "steady_s": solve.steady_s,
            "residual_norm": solve.residual_norm,
            "stationarity_norm": solve.stationarity_norm,
            "boundary_residual_norm": solve.boundary_norm,
            "score": 0.0,
        }
        row["score"] = _score(row)
        rows.append(row)

    best = min(rows, key=_score) if rows else {}
    recovered = bool(best and best["success"] and best["stationarity_norm"] <= 1.0e-3)
    if not recovered:
        _append_jsonl(
            failures,
            {
                "ts": time.time(),
                "case": args.case,
                "failure_class": "F1_baseline_unstable",
                "evidence": best,
            },
        )

    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "case": args.case,
        "classification": "recovered" if recovered else "not-recovered",
        "best": best,
        "row_count": len(rows),
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
