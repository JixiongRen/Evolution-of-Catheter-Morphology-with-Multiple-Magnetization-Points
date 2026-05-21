#!/usr/bin/env python3
"""Inspect JAX device setup and magnetic-model integration."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import json

from single_layer_ik.jax_config import configure_jax_enable_x64

configure_jax_enable_x64()

def check_gpu_availability():
    """Print basic JAX device information."""
    print("=" * 60)
    print("GPU and device diagnostics")
    print("=" * 60)
    
    devices = jax.devices()
    print(f"\nAvailable devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.device_kind} - {device}")
    
    print(f"\nDefault backend: {jax.default_backend()}")
    
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    print(f"\nTest computation finished. Result shape: {y.shape}")
    print(f"Compute device: {y.device()}" if hasattr(y, '__call__') else f"Compute device: {y.devices()}")
    
    return devices


def check_magnetic_model_integration():
    """Verify that the magnetic modules can be imported and configured."""
    print("\n" + "=" * 60)
    print("Magnetic-model integration diagnostics")
    print("=" * 60)
    
    from .forward_kinematics_optimized_bak.fk import build_solver_params
    from .forward_kinematics_optimized_bak.equilibrium_solver_nondim import residual_bar, with_coil_currents
    
    actuation_table_path = Path("supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl")
    if actuation_table_path.exists():
        print(f"\nActuation table found: {actuation_table_path}")
    else:
        print(f"\nActuation table not found: {actuation_table_path}")
        return False
    
    try:
        flex_lengths_tail = [0.03, 0.03]
        rigid_lengths = [0.003, 0.003, 0.003]
        M_list = [3, 2, 2]
        
        m_body_list = [[0, 0, 0.005301], [0, 0, 0.005301], [0, 0, 0.005301]]
        
        print("\nChecking solver parameter inputs...")
        print(f"  flex_lengths_tail: {flex_lengths_tail}")
        print(f"  rigid_lengths: {rigid_lengths}")
        print(f"  M_list: {M_list}")
        print(f"  m_body_list: {m_body_list}")
        
        print("\nMagnetic-model modules imported successfully.")
        
    except Exception as e:
        print(f"\nMagnetic-model integration check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def analyze_experiment_traces():
    """Compare trace files from two representative IK runs."""
    print("\n" + "=" * 60)
    print("Experiment trace analysis")
    print("=" * 60)
    
    pipeline_dir = Path("ik_out/diag_mag_realistic_t2_auto_pipeline")
    exp2_dir = Path("ik_out/diag_mag_realistic_t2_auto_init_exp2")
    
    if not pipeline_dir.exists() or not exp2_dir.exists():
        print("\nRequired experiment directories are missing.")
        return
    
    def read_trace(trace_path):
        trace = []
        with open(trace_path, 'r') as f:
            for line in f:
                if line.strip():
                    trace.append(json.loads(line))
        return trace
    
    pipeline_trace = read_trace(pipeline_dir / "trace.jsonl")
    exp2_trace = read_trace(exp2_dir / "trace.jsonl")
    
    print(f"\nPipeline trace length: {len(pipeline_trace)}")
    print(f"Exp2 trace length: {len(exp2_trace)}")
    
    print("\n--- Pipeline current evolution ---")
    for idx in [0, 10, 50, 100, 200, len(pipeline_trace)-1]:
        if idx < len(pipeline_trace):
            t = pipeline_trace[idx]
            I_A = t.get('I_A', [0]*8)
            I_max = max([abs(x) for x in I_A])
            I_norm = np.linalg.norm(I_A)
            print(f"iter {t.get('iter', idx):4d}: I_max={I_max:6.2f}A, I_norm={I_norm:6.2f}A, dist={t.get('dist_to_target_mm', 0):7.2f}mm")
    
    print("\n--- Exp2 current evolution ---")
    for idx in [0, 10, 50, 100, len(exp2_trace)-1]:
        if idx < len(exp2_trace):
            t = exp2_trace[idx]
            I_A = t.get('I_A', [0]*8)
            I_max = max([abs(x) for x in I_A])
            I_norm = np.linalg.norm(I_A)
            print(f"iter {t.get('iter', idx):4d}: I_max={I_max:6.2f}A, I_norm={I_norm:6.2f}A, dist={t.get('dist_to_target_mm', 0):7.2f}mm")
    
    print("\n--- Pipeline equilibrium residuals ---")
    for idx in [0, 50, 100, 200, len(pipeline_trace)-1]:
        if idx < len(pipeline_trace):
            t = pipeline_trace[idx]
            print(f"iter {t.get('iter', idx):4d}: norm_E={t.get('norm_E_raw', 0):8.4f}, normR={t.get('normR', 0):8.4f}")
    
    print("\n--- Exp2 equilibrium residuals ---")
    for idx in [0, 50, 100, len(exp2_trace)-1]:
        if idx < len(exp2_trace):
            t = exp2_trace[idx]
            print(f"iter {t.get('iter', idx):4d}: norm_E={t.get('norm_E_raw', 0):8.4f}, normR={t.get('normR', 0):8.4f}")


def main():
    """Run all diagnostics."""
    print("\n" + "=" * 60)
    print("IK diagnostic tool")
    print("=" * 60)
    
    devices = check_gpu_availability()
    
    mag_ok = check_magnetic_model_integration()
    
    analyze_experiment_traces()
    
    print("\n" + "=" * 60)
    print("Diagnostic summary")
    print("=" * 60)
    print(f"GPU device count: {len(devices)}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Magnetic-model integration: {'ok' if mag_ok else 'failed'}")
    print("\nSuggestions:")
    if jax.default_backend() == 'cpu':
        print("  CPU backend is active. Configure GPU acceleration if available.")
    if not mag_ok:
        print("  Magnetic-model integration requires further inspection.")
    print("=" * 60)


if __name__ == "__main__":
    main()
