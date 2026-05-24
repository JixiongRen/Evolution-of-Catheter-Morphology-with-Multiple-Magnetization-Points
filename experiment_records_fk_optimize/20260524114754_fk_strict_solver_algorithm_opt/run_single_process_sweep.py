"""Run fixed-case FK solver candidates in one Python process."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.efficiency_optimization.fk_under_60s_driver import build_parser, run_continuation


EXP_DIR = Path(__file__).resolve().parent


def make_args(case_name: str, extra: list[str]):
    base = [
        "--out_dir",
        str(EXP_DIR / case_name),
        "--case_name",
        case_name,
        "--mode",
        "continuation",
        "--classification",
        "speed-only result",
        "--scale_mode",
        "optimized",
        "--scale_include_external",
        "--warmup",
        "--tol",
        "1e-5",
        "--damping_mode",
        "identity",
        "--dense_solve_method",
        "cholesky",
        "--current_factors",
        "0,1.0",
        "--stage_max_iter",
        "60",
        "--stage_tol",
        "1e-3",
        "--final_tol",
        "1e-5",
    ]
    return build_parser().parse_args(base + extra)


def main() -> int:
    candidates = [
        ("sweep_cholesky_lm01_final1400", ["--lm_damping", "0.1", "--final_max_iter", "1400"]),
        ("sweep_cholesky_lm003_final1400", ["--lm_damping", "0.03", "--final_max_iter", "1400"]),
        ("sweep_cholesky_lm1_final1400", ["--lm_damping", "1.0", "--final_max_iter", "1400"]),
        (
            "sweep_cholesky_trust_eta001_final1400",
            [
                "--lm_damping",
                "0.3",
                "--trust_region_eta",
                "0.01",
                "--max_damping_attempts",
                "3",
                "--final_max_iter",
                "1400",
            ],
        ),
        (
            "sweep_cholesky_trust_eta0001_final1400",
            [
                "--lm_damping",
                "0.3",
                "--trust_region_eta",
                "0.0001",
                "--max_damping_attempts",
                "3",
                "--final_max_iter",
                "1400",
            ],
        ),
    ]
    rows = []
    for case_name, extra in candidates:
        start = time.perf_counter()
        args = make_args(case_name, extra)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = run_continuation(args, out_dir)
        summary["process_wall_s"] = time.perf_counter() - start
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(
            {
                "case_name": case_name,
                "ok": bool(summary["ok"]),
                "wall_s": float(summary["wall_s"]),
                "final_normE": float(summary["final_normE"]),
                "n_iter": summary["n_iter"],
                "process_wall_s": float(summary["process_wall_s"]),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (EXP_DIR / "single_process_sweep_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
