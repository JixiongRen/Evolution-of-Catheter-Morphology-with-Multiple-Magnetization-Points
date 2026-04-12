from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run comparable single-layer IK experiments.")
    p.add_argument("--config", type=str, required=True, help="Experiment JSON config")
    p.add_argument("--python_exe", type=str, default=sys.executable, help="Python executable to launch child runs")
    p.add_argument("--out_root", type=str, default=None, help="Output directory root for experiment runs")
    p.add_argument("--plot_max_lines", type=int, default=200, help="Max centerlines in generated 3D plots")
    p.add_argument("--cpu_only", action="store_true", help="Force JAX_PLATFORMS=cpu in child runs")
    return p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trace_stats(path: Path) -> Dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return {
            "n_trace_records": 0,
            "low_rho_ratio": None,
            "reject_ratio": None,
            "mean_backtrack_count": None,
            "mean_norm_E_raw": None,
            "mean_smooth_metric_raw": None,
            "final_trace": None,
        }

    low_rho = sum(1 for r in records if float(r.get("rho", 0.0)) <= 0.25)
    rejected = sum(1 for r in records if not bool(r.get("accepted", False)))
    mean_backtrack = sum(int(r.get("backtrack_count", 0)) for r in records) / max(len(records), 1)
    mean_norm_E_raw = sum(float(r.get("norm_E_raw", 0.0)) for r in records) / max(len(records), 1)
    mean_smooth_metric_raw = sum(float(r.get("smooth_metric_raw", 0.0)) for r in records) / max(len(records), 1)

    return {
        "n_trace_records": len(records),
        "low_rho_ratio": low_rho / len(records),
        "reject_ratio": rejected / len(records),
        "mean_backtrack_count": mean_backtrack,
        "mean_norm_E_raw": mean_norm_E_raw,
        "mean_smooth_metric_raw": mean_smooth_metric_raw,
        "final_trace": records[-1],
    }


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


def _aggregate_by_variant(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["variant"]), []).append(record)

    summary: Dict[str, Any] = {}
    for variant, items in grouped.items():
        success_count = sum(1 for x in items if bool(x.get("result", {}).get("success", False)))
        pos_err = [
            float(x["result"]["diag"]["pos_err_norm_mm"])
            for x in items
            if isinstance(x.get("result"), dict) and isinstance(x["result"].get("diag"), dict) and x["result"]["diag"].get("pos_err_norm_mm") is not None
        ]
        norm_e = [
            float(x["result"]["diag"]["norm_E_raw"])
            for x in items
            if isinstance(x.get("result"), dict) and isinstance(x["result"].get("diag"), dict) and x["result"]["diag"].get("norm_E_raw") is not None
        ]
        n_iter = [
            float(x["result"]["n_iter"])
            for x in items
            if isinstance(x.get("result"), dict) and x["result"].get("n_iter") is not None
        ]
        reject_ratio = [
            float(x["trace_stats"]["reject_ratio"])
            for x in items
            if isinstance(x.get("trace_stats"), dict) and x["trace_stats"].get("reject_ratio") is not None
        ]
        low_rho_ratio = [
            float(x["trace_stats"]["low_rho_ratio"])
            for x in items
            if isinstance(x.get("trace_stats"), dict) and x["trace_stats"].get("low_rho_ratio") is not None
        ]
        smooth_metric = [
            float(x["trace_stats"]["mean_smooth_metric_raw"])
            for x in items
            if isinstance(x.get("trace_stats"), dict) and x["trace_stats"].get("mean_smooth_metric_raw") is not None
        ]

        summary[variant] = {
            "n_cases": len(items),
            "success_count": int(success_count),
            "success_rate": float(success_count / max(len(items), 1)),
            "mean_pos_err_norm_mm": _mean_or_none(pos_err),
            "mean_norm_E_raw": _mean_or_none(norm_e),
            "mean_n_iter": _mean_or_none(n_iter),
            "mean_reject_ratio": _mean_or_none(reject_ratio),
            "mean_low_rho_ratio": _mean_or_none(low_rho_ratio),
            "mean_smooth_metric_raw": _mean_or_none(smooth_metric),
        }
    return summary


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
    variants = cfg.get("variants", [])
    if not base_args:
        raise ValueError("config.base_args is empty")
    if not cases:
        raise ValueError("config.cases is empty")
    if not variants:
        raise ValueError("config.variants is empty")

    summary: Dict[str, Any] = {
        "config": str(cfg_path),
        "python_exe": args.python_exe,
        "out_root": str(out_root),
        "base_args": base_args,
        "cases": cases,
        "variants": variants,
        "runs": [],
    }

    child_env = dict(os.environ)
    if args.cpu_only:
        child_env["JAX_PLATFORMS"] = "cpu"

    for case in cases:
        case_name = str(case["name"])
        case_args = case.get("args", {})
        for variant in variants:
            variant_name = str(variant["name"])
            merged_args = dict(base_args)
            merged_args.update(case_args)
            merged_args.update(variant.get("args", {}))

            run_dir = out_root / case_name / variant_name
            cmd = [args.python_exe, "-m", "single_layer_ik.run_joint_ik"] + _flatten_args(merged_args, run_dir=run_dir)
            print(f"[experiment] running case={case_name} variant={variant_name}")
            print("[experiment] command:", " ".join(cmd))
            completed = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=child_env,
                capture_output=True,
                text=True,
                check=False,
            )

            record: Dict[str, Any] = {
                "case": case_name,
                "variant": variant_name,
                "description": variant.get("description", ""),
                "command": cmd,
                "returncode": int(completed.returncode),
                "stdout_tail": completed.stdout.splitlines()[-20:],
                "stderr_tail": completed.stderr.splitlines()[-20:],
                "run_dir": str(run_dir),
            }

            if run_dir.exists():
                plot_cmd = [
                    args.python_exe,
                    "-m",
                    "single_layer_ik.plot_joint_ik_run",
                    f"--run_dir={run_dir}",
                    f"--max_lines={int(args.plot_max_lines)}",
                ]
                plot_completed = subprocess.run(
                    plot_cmd,
                    cwd=str(repo_root),
                    env=child_env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                record["plot_returncode"] = int(plot_completed.returncode)
                record["plot_stdout_tail"] = plot_completed.stdout.splitlines()[-20:]
                record["plot_stderr_tail"] = plot_completed.stderr.splitlines()[-20:]

            result_path = run_dir / "result.json"
            trace_path = run_dir / "trace.jsonl"
            if result_path.exists():
                record["result"] = _read_json(result_path)
            if trace_path.exists():
                record["trace_stats"] = _read_trace_stats(trace_path)

            summary["runs"].append(record)

    summary["aggregates"] = _aggregate_by_variant(summary["runs"])

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[experiment] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
