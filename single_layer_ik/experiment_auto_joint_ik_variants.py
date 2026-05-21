from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run comparable single-layer IK auto-init experiments.")
    p.add_argument("--config", type=str, required=True, help="Experiment JSON config")
    p.add_argument("--python_exe", type=str, default=sys.executable, help="Python executable to launch child runs")
    p.add_argument("--out_root", type=str, default=None, help="Output directory root for experiment runs")
    p.add_argument("--cpu_only", action="store_true", help="Force JAX_PLATFORMS=cpu in child runs")
    p.add_argument("--skip_existing", action="store_true", help="Skip cases that already contain result.json")
    return p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _flatten_args(args: Dict[str, Any], *, run_dir: Path) -> list[str]:
    merged = dict(args)
    merged["out_dir"] = str(run_dir)

    flat: list[str] = []
    for key, value in merged.items():
        if value is None:
            continue
        if value is True:
            flat.append(f"--{key}")
        elif value is False:
            continue
        else:
            flat.append(f"--{key}={value}")
    return flat


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    diag = result.get("diag", {}) if isinstance(result, dict) else {}
    return {
        "success": bool(result.get("success", False)),
        "stop_reason": result.get("stop_reason"),
        "n_iter": result.get("n_iter"),
        "pos_err_norm_mm": diag.get("pos_err_norm_mm"),
        "max_interface_gap_mm": diag.get("max_interface_gap_mm"),
        "norm_E_raw": diag.get("norm_E_raw"),
        "smooth_metric_raw": diag.get("smooth_metric_raw"),
        "max_current_utilization": diag.get("max_current_utilization"),
    }


def _aggregate(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    done = [r for r in records if isinstance(r.get("result"), dict)]
    success = [r for r in done if bool(r["result"].get("success", False))]
    pos_err = [
        float(r["result"]["diag"]["pos_err_norm_mm"])
        for r in done
        if isinstance(r["result"].get("diag"), dict) and r["result"]["diag"].get("pos_err_norm_mm") is not None
    ]
    max_gap = [
        float(r["result"]["diag"]["max_interface_gap_mm"])
        for r in done
        if isinstance(r["result"].get("diag"), dict) and r["result"]["diag"].get("max_interface_gap_mm") is not None
    ]
    n_iter = [
        float(r["result"]["n_iter"])
        for r in done
        if r["result"].get("n_iter") is not None
    ]
    current_util = [
        float(r["result"]["diag"]["max_current_utilization"])
        for r in done
        if isinstance(r["result"].get("diag"), dict) and r["result"]["diag"].get("max_current_utilization") is not None
    ]
    return {
        "n_runs": len(records),
        "n_completed": len(done),
        "n_success": len(success),
        "success_rate": float(len(success) / max(len(records), 1)),
        "mean_pos_err_norm_mm": _mean_or_none(pos_err),
        "mean_max_interface_gap_mm": _mean_or_none(max_gap),
        "mean_n_iter": _mean_or_none(n_iter),
        "mean_max_current_utilization": _mean_or_none(current_util),
    }


def main() -> None:
    args = build_argparser().parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    cfg = _read_json(cfg_path)

    if args.out_root is None:
        out_root = repo_root / "ik_out" / cfg_path.stem
    else:
        out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base_args: Dict[str, Any] = cfg.get("base_args", {})
    cases = cfg.get("cases", [])
    if not base_args:
        raise ValueError("config.base_args is empty")
    if not cases:
        raise ValueError("config.cases is empty")

    summary: Dict[str, Any] = {
        "config": str(cfg_path),
        "python_exe": args.python_exe,
        "out_root": str(out_root),
        "base_args": base_args,
        "cases": cases,
        "runs": [],
    }

    child_env = dict(os.environ)
    if args.cpu_only:
        child_env["JAX_PLATFORMS"] = "cpu"

    for case in cases:
        case_name = str(case["name"])
        merged_args = dict(base_args)
        merged_args.update(case.get("args", {}))
        run_dir = out_root / case_name
        result_path = run_dir / "result.json"

        if args.skip_existing and result_path.exists():
            record: Dict[str, Any] = {
                "case": case_name,
                "description": case.get("description", ""),
                "run_dir": str(run_dir),
                "skipped": True,
                "result": _read_json(result_path),
            }
            record["summary"] = _summarize_result(record["result"])
            summary["runs"].append(record)
            continue

        cmd = [args.python_exe, "-m", "single_layer_ik.auto_run_joint_ik"] + _flatten_args(merged_args, run_dir=run_dir)
        t0 = time.time()
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_s = time.time() - t0

        record = {
            "case": case_name,
            "description": case.get("description", ""),
            "command": cmd,
            "returncode": int(completed.returncode),
            "elapsed_s": float(elapsed_s),
            "stdout_tail": completed.stdout.splitlines()[-30:],
            "stderr_tail": completed.stderr.splitlines()[-30:],
            "run_dir": str(run_dir),
        }
        if result_path.exists():
            record["result"] = _read_json(result_path)
            record["summary"] = _summarize_result(record["result"])
        summary["runs"].append(record)

        summary_path = out_root / "summary.json"
        summary["aggregates"] = _aggregate(summary["runs"])
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["aggregates"] = _aggregate(summary["runs"])
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[experiment] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
