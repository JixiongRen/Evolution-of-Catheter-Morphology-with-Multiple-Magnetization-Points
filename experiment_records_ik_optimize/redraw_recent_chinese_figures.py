"""Redraw recent IK experiment figures with CJK font support."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


BASE_DIR = ROOT / "experiment_records_ik_optimize"


def _load_module(path: Path) -> ModuleType:
    name = path.stem + "_" + path.parent.name
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _summary_rows(exp_dir: Path) -> list[dict]:
    data = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    return list(data["case_summaries"])


def _redraw_p2c15(exp_dir: Path) -> list[Path]:
    module = _load_module(exp_dir / "run_p2c15_slack_suppressed_sensitivity_audit.py")
    rows = _summary_rows(exp_dir)
    module._plot_summary(rows)
    outputs = [exp_dir / "sensitivity_gate_summary.png"]
    for row in rows:
        eps = float(row["eps_alpha"])
        npz = np.load(exp_dir / f"alpha_{row['alpha']:.2f}_eps_{eps:.3f}_sensitivity_audit.npz")
        centerlines = {
            "base": npz["base_centerline"],
            "plus": npz["plus_centerline"],
            "minus": npz["minus_centerline"],
        }
        module._plot_centerlines(eps, centerlines)
        outputs.append(exp_dir / f"centerline_reoptimized_eps_{eps:.3f}.png")
    return outputs


def _redraw_generic(exp_dir: Path, script_name: str, expected: list[str]) -> list[Path]:
    module = _load_module(exp_dir / script_name)
    module._plot(_summary_rows(exp_dir))
    return [exp_dir / name for name in expected]


def main() -> None:
    font = configure_matplotlib_cjk()
    targets = [
        _redraw_p2c15(
            BASE_DIR / "20260605125315_p2c15_slack_suppressed_sensitivity_audit"
        ),
        _redraw_generic(
            BASE_DIR / "20260605132600_p2c16_alpha_boundary_phase_map",
            "run_p2c16_alpha_boundary_phase_map.py",
            [
                "alpha_boundary_phase_map.png",
                "alpha_classification_strip.png",
                "baseline_centerline_by_alpha.png",
            ],
        ),
        _redraw_generic(
            BASE_DIR / "20260605145500_p2c17_refined_boundary_path_dependence",
            "run_p2c17_refined_boundary_path_dependence.py",
            [
                "refined_boundary_metrics.png",
                "refined_boundary_classification.png",
                "initialization_gate_stack.png",
            ],
        ),
        _redraw_generic(
            BASE_DIR / "20260605153000_p2c18_lower_alpha_breakpoint_load_audit",
            "run_p2c18_lower_alpha_breakpoint_load_audit.py",
            [
                "lower_breakpoint_metrics.png",
                "lower_breakpoint_initialization_gates.png",
            ],
        ),
        _redraw_generic(
            BASE_DIR / "20260606103000_p2c19_ultrafine_lower_alpha_breakpoint",
            "run_p2c19_ultrafine_lower_alpha_breakpoint.py",
            [
                "lower_breakpoint_metrics.png",
                "lower_breakpoint_initialization_gates.png",
            ],
        ),
    ]
    outputs = [path for group in targets for path in group]
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise RuntimeError("Missing redrawn figures: " + ", ".join(missing))
    print(json.dumps({"font": font, "redrawn": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
