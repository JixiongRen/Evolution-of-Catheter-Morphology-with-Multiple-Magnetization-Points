from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .forward_kinematics_optimized_bak.fk_engine import ForwardKinematicsEngine
from .joint_problem import extract_centerline_p_dim


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=np.float64)


def _resample_polyline(points: np.ndarray, n: int = 240) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= 1:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    grid = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(grid, s, pts[:, i]) for i in range(3)])


def _centerline_rms_mm(a: np.ndarray, b: np.ndarray) -> float:
    aa = _resample_polyline(a)
    bb = _resample_polyline(b)
    return float(np.sqrt(np.mean(np.sum((aa - bb) ** 2, axis=1))) * 1e3)


def _engine_from_config(cfg: dict[str, Any], *, max_iter: int, tol: float, lm_damping: float) -> ForwardKinematicsEngine:
    geometry = cfg["geometry"]
    materials = cfg["materials"]
    loads = cfg["loads"]
    solver = cfg.get("solver", {})
    return ForwardKinematicsEngine(
        flex_lengths_tail=geometry["flex_lengths_tail"],
        rigid_lengths=geometry["rigid_lengths"],
        M_list=geometry["M_list"],
        L1_min=float(geometry["L1_min"]),
        flex_d_outer=materials["flex_d_outer"],
        flex_E=materials["flex_E"],
        flex_G=materials["flex_G"],
        flex_rho=materials["flex_rho"],
        rigid_d_outer=materials["rigid_d_outer"],
        rigid_rho=materials["rigid_rho"],
        p0_dim=geometry["p0_dim"],
        Q0_wxyz=geometry["Q0_wxyz"],
        axis_body=geometry["axis_body"],
        enable_gravity=bool(loads["enable_gravity"]),
        g_world=loads["g_world"],
        enable_magnetics=bool(loads["enable_magnetics"]),
        calib_file=loads.get("calib_file"),
        actuation_table_pkl=loads.get("actuation_table_pkl"),
        m_body_list=loads.get("m_body_list"),
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=float(lm_damping),
        jac_method=str(solver.get("jac_method", "fwd")),
        L_protrude_max=float(geometry["L_protrude_max"]),
    )


def _evaluate_one(
    *,
    cfg: dict[str, Any],
    state: dict[str, np.ndarray],
    mode: str,
    override: np.ndarray | None,
    max_iter: int,
    tol: float,
    lm_damping: float,
    centerline_rigid_samples: int,
) -> tuple[dict[str, Any], np.ndarray]:
    engine = _engine_from_config(cfg, max_iter=max_iter, tol=tol, lm_damping=lm_damping)
    z_fk, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=state["I_star"],
        L_protrude=float(state["x_star"][0]),
        warm_start=False,
        override_z0_bar=None if override is None else jnp.asarray(override, dtype=jnp.float64),
        return_stats=True,
        stats_include_jacobian=False,
        verbose=False,
    )
    centerline_fk = _to_numpy(
        extract_centerline_p_dim(
            z_bar=z_fk,
            scales=params.scales,
            M_list=params.M_list,
            include_rigid=True,
            rigid_sample_count=int(centerline_rigid_samples),
        )
    )
    ik_tip = np.asarray(state["p_tip_ik"], dtype=np.float64).reshape(3,)
    fk_tip = centerline_fk[-1]
    p_des = np.asarray(state["p_des"], dtype=np.float64).reshape(3,)
    ik_target_mm = float(np.linalg.norm(ik_tip - p_des) * 1e3)
    fk_target_mm = float(np.linalg.norm(fk_tip - p_des) * 1e3)
    row = {
        "mode": mode,
        "fk_ok_flag": bool(ok),
        "fk_stop_reason": "" if stats is None else str(getattr(stats, "stop_reason", "")),
        "fk_n_iter": -1 if stats is None else int(getattr(stats, "n_iter", -1)),
        "fk_normE": float("nan") if stats is None else float(getattr(stats, "normE", float("nan"))),
        "ik_target_err_mm": ik_target_mm,
        "fk_target_err_mm": fk_target_mm,
        "target_err_delta_mm": abs(fk_target_mm - ik_target_mm),
        "ik_fk_tip_mm": float(np.linalg.norm(fk_tip - ik_tip) * 1e3),
        "ik_fk_centerline_rms_mm": _centerline_rms_mm(centerline_fk, state["centerline_ik"]),
    }
    row["pass"] = bool(
        row["fk_normE"] <= tol
        and row["ik_fk_tip_mm"] < 0.1
        and row["ik_fk_centerline_rms_mm"] < 0.1
        and row["target_err_delta_mm"] < 0.05
    )
    return row, centerline_fk


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict FK round-trip verification for single-layer IK outputs.")
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--max_iter", type=int, default=6000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--lm_damping", type=float, default=1e-2)
    p.add_argument("--perturb_repeats", type=int, default=3)
    p.add_argument("--perturb_std", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--short_circuit_on_fail", action="store_true")
    return p


FIELDNAMES = [
    "mode",
    "pass",
    "fk_ok_flag",
    "fk_stop_reason",
    "fk_n_iter",
    "fk_normE",
    "ik_target_err_mm",
    "fk_target_err_mm",
    "target_err_delta_mm",
    "ik_fk_tip_mm",
    "ik_fk_centerline_rms_mm",
]


def _classify(rows: list[dict[str, Any]], *, complete: bool) -> str:
    if not rows:
        return "missing_roundtrip"
    inherited = next((r for r in rows if r["mode"] == "state_inherited"), None)
    if inherited is None:
        return "missing_roundtrip"
    if not bool(inherited["pass"]):
        return "rejected"
    if complete and all(bool(r["pass"]) for r in rows):
        return "main_result_usable"
    if complete:
        return "path_dependent"
    return "running_inherited_passed"


def _write_outputs(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    centerline_payload: dict[str, np.ndarray],
    classification: str,
    args: argparse.Namespace,
    complete: bool,
) -> None:
    with (run_dir / "fk_roundtrip_summary.partial.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(run_dir / "fk_roundtrip_centerlines.partial.npz", **centerline_payload)
    payload = {
        "classification": classification,
        "complete": bool(complete),
        "criteria": {
            "fk_normE_max": float(args.tol),
            "ik_fk_tip_mm_max": 0.1,
            "ik_fk_centerline_rms_mm_max": 0.1,
            "target_err_delta_mm_max": 0.05,
        },
        "rows": rows,
    }
    (run_dir / "fk_roundtrip_summary.partial.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    if complete:
        with (run_dir / "fk_roundtrip_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(run_dir / "fk_roundtrip_centerlines.npz", **centerline_payload)
        (run_dir / "fk_roundtrip_summary.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    args = build_argparser().parse_args()
    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / "config_full.json"
    state_path = run_dir / "state_final.npz"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    cfg = _read_json(cfg_path)
    state_npz = np.load(state_path)
    state = {name: np.asarray(state_npz[name], dtype=np.float64) for name in state_npz.files}
    centerline_samples = int(cfg.get("diagnostics", {}).get("centerline_rigid_samples", 2))
    z_star = np.asarray(state["z_star_bar"], dtype=np.float64).reshape(-1)

    rng = np.random.default_rng(int(args.seed))
    jobs: list[tuple[str, np.ndarray | None]] = [("state_inherited", z_star), ("cold_start", None)]
    for idx in range(int(args.perturb_repeats)):
        jobs.append((f"perturb_{idx + 1}", z_star + rng.normal(0.0, float(args.perturb_std), size=z_star.shape)))

    rows: list[dict[str, Any]] = []
    centerline_payload: dict[str, np.ndarray] = {"ik_centerline": state["centerline_ik"]}
    complete = False
    for mode, override in jobs:
        print(f"[FK roundtrip] start mode={mode}", flush=True)
        row, centerline_fk = _evaluate_one(
            cfg=cfg,
            state=state,
            mode=mode,
            override=override,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            lm_damping=float(args.lm_damping),
            centerline_rigid_samples=centerline_samples,
        )
        rows.append(row)
        centerline_payload[f"{mode}_centerline"] = centerline_fk
        print(
            "[FK roundtrip] done mode={mode} pass={passed} normE={normE:.3e} "
            "tip={tip:.4g}mm rms={rms:.4g}mm".format(
                mode=mode,
                passed=row["pass"],
                normE=row["fk_normE"],
                tip=row["ik_fk_tip_mm"],
                rms=row["ik_fk_centerline_rms_mm"],
            ),
            flush=True,
        )
        classification = _classify(rows, complete=False)
        _write_outputs(
            run_dir=run_dir,
            rows=rows,
            centerline_payload=centerline_payload,
            classification=classification,
            args=args,
            complete=False,
        )
        if args.short_circuit_on_fail and not bool(row["pass"]):
            break
    else:
        complete = True

    classification = _classify(rows, complete=complete or bool(args.short_circuit_on_fail))
    _write_outputs(
        run_dir=run_dir,
        rows=rows,
        centerline_payload=centerline_payload,
        classification=classification,
        args=args,
        complete=True,
    )
    print(f"[FK roundtrip] {run_dir.name}: {classification}")


if __name__ == "__main__":
    main()
