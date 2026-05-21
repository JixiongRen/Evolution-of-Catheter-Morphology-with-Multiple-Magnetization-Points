#!/usr/bin/env python3
"""Compare IK behavior under different magnetization settings."""
from __future__ import annotations

import subprocess
import sys
import json
import os
from pathlib import Path
import numpy as np

def run_ik_experiment(exp_name: str, m_body_list: str, p_des: str = "0.068,-0.031,0.042"):
    """Run a single IK experiment for one magnetization configuration.

    Args:
        exp_name: Experiment name used in the output directory.
        m_body_list: Magnetization vectors passed to the IK pipeline.
        p_des: Target tip position in meters.

    Returns:
        The output directory path on success, or ``None`` on failure.
    """
    
    out_dir = f"ik_out/mag_influence_test_{exp_name}"
    
    cmd = [
        sys.executable,
        "-m",
        "single_layer_ik.auto_run_joint_ik",
        "--p_des", p_des,
        "--m_body_list", m_body_list,
        "--enable_magnetics",
        "--M_list", "3,2,2",
        "--max_iter", "100",
        "--dist_tol_mm", "1.0",
        "--out_dir", out_dir,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"m_body_list: {m_body_list}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}\n")
    
    try:
        env = dict(os.environ)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return out_dir
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def compare_results(exp_dirs: dict):
    """Print a summary table for multiple experiment outputs."""
    print(f"\n{'='*60}")
    print("Experiment comparison")
    print(f"{'='*60}\n")
    
    results = {}
    for name, dir_path in exp_dirs.items():
        if dir_path is None:
            continue
        
        result_file = Path(dir_path) / "result.json"
        if not result_file.exists():
            print(f"Missing result file for {name}")
            continue
        
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        results[name] = {
            'p_tip': result.get('final', {}).get('p_tip_m', result.get('diag', {}).get('p_tip_m', [0,0,0])),
            'I_A': result.get('I_A', [0]*8),
            'x_m': result.get('x_m', 0),
            'pos_err_mm': result.get('diag', {}).get('pos_err_norm_mm', 999),
            'norm_E': result.get('diag', {}).get('norm_E_raw', 999),
            'n_iter': result.get('n_iter', 0),
        }
        
        print(f"{name}:")
        print(f"  Tip position: {results[name]['p_tip']}")
        print(f"  Position error: {results[name]['pos_err_mm']:.4f} mm")
        print(f"  Equilibrium residual: {results[name]['norm_E']:.6f}")
        print(f"  Iterations: {results[name]['n_iter']}")
        print(f"  Insertion depth: {results[name]['x_m']:.6f} m")
        print(f"  Currents: {[f'{x:.2f}' for x in results[name]['I_A']]}")
        print()
    
    if len(results) >= 2:
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name1, name2 = names[i], names[j]
                p1 = np.array(results[name1]['p_tip'])
                p2 = np.array(results[name2]['p_tip'])
                diff = np.linalg.norm(p1 - p2) * 1000
                
                print(f"{name1} vs {name2}:")
                print(f"  Tip position difference: {diff:.4f} mm")
                
                if diff < 1.0:
                    print("  Warning: difference is below 1 mm; the magnetization change may not be active.")
                else:
                    print("  Difference is significant; the magnetization setting affects the solution.")
                print()


def main():
    """Run the comparison workflow."""
    print("\n" + "="*60)
    print("Magnetization influence test")
    print("="*60)
    
    
    experiments = {
        "z_positive": "0,0,0.005301; 0,0,0.005301; 0,0,0.005301",
        "z_negative": "0,0,-0.005301; 0,0,-0.005301; 0,0,-0.005301",
        "y_positive": "0,0.005301,0; 0,0.005301,0; 0,0.005301,0",
    }
    
    exp_dirs = {}
    for name, m_body in experiments.items():
        exp_dirs[name] = run_ik_experiment(name, m_body)
    
    compare_results(exp_dirs)
    
    print("\n" + "="*60)
    print("Test complete")
    print("="*60)


if __name__ == "__main__":
    main()
