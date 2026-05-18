#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate heuristic current initializations from a target position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import jax.numpy as jnp

from supiee_auto_diff.actuation_interpolator_jax import (
    load_actuation_table,
    interpolate_A_checked,
)


def compute_target_direction(p_des: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Compute the normalized direction from the start point to the target.

    Args:
        p_des: Target position.
        p0: Reference start position.

    Returns:
        Normalized direction vector.
    """
    direction = p_des - p0
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return direction / norm


def analyze_actuation_matrix(
    A: np.ndarray,
    target_dir: np.ndarray,
    m_body: np.ndarray = np.array([0.0, 0.0, 0.005301]),
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Estimate coil effectiveness from the local actuation matrix.

    Args:
        A: Local actuation matrix.
        target_dir: Desired bending direction.
        m_body: Reference magnetic moment in the body frame.

    Returns:
        Coil weights, coil effectiveness scores, and diagnostic metadata.
    """
    
    A_Bx = A[0, :]
    
    A_By = A[1, :]  # (8,)
    A_Bz = A[2, :]  # (8,)
    
    
    effectiveness_Bx = np.abs(A_Bx)
    
    
    desired_Bx_sign = np.sign(target_dir[0]) if abs(target_dir[0]) > 0.1 else 0.0
    
    if abs(desired_Bx_sign) > 0.5:
        effectiveness = A_Bx * desired_Bx_sign
        effectiveness = np.maximum(effectiveness, 0.0)
    else:
        effectiveness = effectiveness_Bx
    
    total_eff = np.sum(effectiveness)
    if total_eff < 1e-12:
        coil_weights = np.ones(8) / 8.0
    else:
        coil_weights = effectiveness / total_eff
    
    analysis = {
        "A_Bx": A_Bx.tolist(),
        "A_By": A_By.tolist(),
        "A_Bz": A_Bz.tolist(),
        "effectiveness": effectiveness.tolist(),
        "coil_weights": coil_weights.tolist(),
        "target_direction": target_dir.tolist(),
        "desired_Bx_sign": float(desired_Bx_sign),
        "total_effectiveness": float(total_eff),
    }
    
    return coil_weights, effectiveness, analysis


def generate_current_init(
    p_des: np.ndarray,
    actuation_table_path: str | Path,
    *,
    p0: Optional[np.ndarray] = None,
    I_magnitude: float = 15.0,
    I_max: float = 50.0,
    m_body: Optional[np.ndarray] = None,
    min_coil_fraction: float = 0.05,
    top_k_coils: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Generate an initial current vector for the IK solver.

    Args:
        p_des: Target tip position.
        actuation_table_path: Path to the actuation lookup table.
        p0: Optional reference start position.
        I_magnitude: Target total current magnitude.
        I_max: Per-coil current limit.
        m_body: Optional reference magnetic moment.
        min_coil_fraction: Minimum retained weight per coil.
        top_k_coils: Optional number of active coils to retain.

    Returns:
        The initial current vector and a diagnostics dictionary.
    """
    if p0 is None:
        p0 = np.array([0.0, 0.0, -0.05])
    if m_body is None:
        m_body = np.array([0.0, 0.0, 0.005301])
    
    p_des = np.asarray(p_des, dtype=np.float64).reshape(3,)
    p0 = np.asarray(p0, dtype=np.float64).reshape(3,)
    
    table = load_actuation_table(actuation_table_path, dtype=jnp.float64)
    
    A_at_target = interpolate_A_checked(table, p_des)  # (8, 8)
    
    target_dir = compute_target_direction(p_des, p0)
    
    coil_weights, effectiveness, analysis = analyze_actuation_matrix(
        A_at_target, target_dir, m_body
    )
    
    if top_k_coils is not None and top_k_coils < 8:
        top_k = int(top_k_coils)
        top_indices = np.argsort(effectiveness)[-top_k:]
        mask = np.zeros(8, dtype=bool)
        mask[top_indices] = True
        coil_weights = np.where(mask, coil_weights, 0.0)
        total = np.sum(coil_weights)
        if total > 1e-12:
            coil_weights = coil_weights / total
    
    coil_weights = np.where(coil_weights >= min_coil_fraction, coil_weights, 0.0)
    
    total = np.sum(coil_weights)
    if total < 1e-12:
        max_idx = np.argmax(effectiveness)
        coil_weights = np.zeros(8)
        coil_weights[max_idx] = 1.0
    else:
        coil_weights = coil_weights / total
    
    I_init = coil_weights * I_magnitude
    
    I_init = np.clip(I_init, -I_max, I_max)
    
    info = {
        "p_des": p_des.tolist(),
        "p0": p0.tolist(),
        "target_direction": target_dir.tolist(),
        "I_magnitude": float(I_magnitude),
        "I_max": float(I_max),
        "coil_weights": coil_weights.tolist(),
        "effectiveness": effectiveness.tolist(),
        "I_init": I_init.tolist(),
        "active_coils": int(np.sum(np.abs(I_init) > 0.1)),
        "max_current": float(np.max(np.abs(I_init))),
        "analysis": analysis,
    }
    
    return I_init, info


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial and reference currents from a target position."
    )
    parser.add_argument(
        "--p_des",
        type=str,
        required=True,
        help="Target position as x,y,z in meters.",
    )
    parser.add_argument(
        "--p0",
        type=str,
        default="0,0,-0.05",
        help="Reference start position as x,y,z in meters.",
    )
    parser.add_argument(
        "--actuation_table",
        type=str,
        default="supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl",
        help="Path to the actuation table.",
    )
    parser.add_argument(
        "--I_magnitude",
        type=float,
        default=15.0,
        help="Target total current magnitude in amperes.",
    )
    parser.add_argument(
        "--I_max",
        type=float,
        default=50.0,
        help="Per-coil current limit in amperes.",
    )
    parser.add_argument(
        "--min_coil_fraction",
        type=float,
        default=0.05,
        help="Minimum retained coil weight.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Keep only the top-k most effective coils.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path.",
    )
    
    args = parser.parse_args()
    
    p_des_vals = [float(x.strip()) for x in args.p_des.split(",")]
    if len(p_des_vals) != 3:
        raise ValueError(f"p_des must contain 3 values, got {len(p_des_vals)}")
    p_des = np.array(p_des_vals)
    
    p0_vals = [float(x.strip()) for x in args.p0.split(",")]
    if len(p0_vals) != 3:
        raise ValueError(f"p0 must contain 3 values, got {len(p0_vals)}")
    p0 = np.array(p0_vals)
    
    I_init, info = generate_current_init(
        p_des=p_des,
        actuation_table_path=args.actuation_table,
        p0=p0,
        I_magnitude=args.I_magnitude,
        I_max=args.I_max,
        min_coil_fraction=args.min_coil_fraction,
        top_k_coils=args.top_k,
    )
    
    output_data = {
        "I_init": I_init.tolist(),
        "I_ref": I_init.tolist(),
        "info": info,
    }
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved results to: {output_path}")
    else:
        print(json.dumps(output_data, indent=2))
    
    print("\n=== Initial current summary ===")
    print(f"Target point: {p_des}")
    print(f"Reference point: {p0}")
    print(f"I_init: {I_init}")
    print(f"Active coils: {info['active_coils']}")
    print(f"Max current: {info['max_current']:.2f} A")
    print(f"Coil weights: {np.array(info['coil_weights'])}")
    
    print("\nPer-coil current allocation:")
    for i, (current, weight) in enumerate(zip(I_init, info['coil_weights'])):
        if abs(current) > 0.01:
            print(f"  Coil {i}: {current:6.2f} A (weight: {weight:.3f})")


if __name__ == "__main__":
    main()
