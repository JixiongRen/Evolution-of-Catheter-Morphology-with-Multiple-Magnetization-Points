#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成基于目标点的初始电流 I_init 和参考电流 I_ref

核心思想：
1. 在目标点 p_des 处计算局部磁场梯度矩阵 A(P)
2. 分析 A(P) 的结构，找出对产生期望磁场方向最有效的线圈组合
3. 生成分布式的初始电流，避免只激活单一线圈

算法原理：
- 磁场与电流的关系：B = A(P) @ I，其中 A(P) 是 8x8 矩阵
- 对于 x-z 平面内的目标，主要需要 B_x 分量（产生 τ_y = m_z * B_x）
- 提取 A(P) 的第一行（B_x 对应行），分析各线圈对 B_x 的贡献
- 根据贡献度分配初始电流
"""

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
    """
    计算从初始点到目标点的方向向量
    
    Args:
        p_des: 目标点 (3,)
        p0: 初始点 (3,)
    
    Returns:
        归一化的方向向量 (3,)
    """
    direction = p_des - p0
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])  # 默认向下
    return direction / norm


def analyze_actuation_matrix(
    A: np.ndarray,
    target_dir: np.ndarray,
    m_body: np.ndarray = np.array([0.0, 0.0, 0.005301]),
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    分析局部磁场梯度矩阵 A(P)，确定最优线圈组合
    
    Args:
        A: 磁场梯度矩阵 (8, 8)，A @ I = [B_x, B_y, B_z, dB_x/dx, dB_y/dy, dB_z/dz, ...]
        target_dir: 目标方向 (3,)
        m_body: 磁矩向量 (3,)，默认为 [0, 0, 0.005301]
    
    Returns:
        coil_weights: 各线圈的权重 (8,)
        effectiveness: 各线圈的有效性得分 (8,)
        analysis: 分析结果字典
    """
    # A 的前3行对应 [B_x, B_y, B_z]
    # 对于 m_body = [0, 0, m_z]，磁力矩 τ = m × B
    # τ_x = m_y * B_z - m_z * B_y = -m_z * B_y
    # τ_y = m_z * B_x - m_x * B_z = m_z * B_x
    # τ_z = m_x * B_y - m_y * B_x = 0
    
    # 对于 x-z 平面内的运动，主要需要 τ_y，即 B_x 分量
    # 提取 A 的第一行（B_x 对 I 的响应）
    A_Bx = A[0, :]  # (8,)，表示各线圈对 B_x 的贡献
    
    # 同时考虑 B_y 和 B_z（用于更全面的分析）
    A_By = A[1, :]  # (8,)
    A_Bz = A[2, :]  # (8,)
    
    # 计算期望的磁力矩方向
    # 对于向目标方向弯曲，需要产生合适的力矩
    # 简化假设：主要关注 B_x（产生 τ_y）
    
    # 方法1：基于 B_x 的贡献度
    effectiveness_Bx = np.abs(A_Bx)
    
    # 方法2：考虑目标方向，计算综合有效性
    # 如果目标在 +x 方向，需要正的 τ_y，即正的 B_x
    # 如果目标在 -x 方向，需要负的 τ_y，即负的 B_x
    
    # 计算期望的 B_x 符号（基于目标方向）
    desired_Bx_sign = np.sign(target_dir[0]) if abs(target_dir[0]) > 0.1 else 0.0
    
    # 如果有明确的方向偏好，考虑符号
    if abs(desired_Bx_sign) > 0.5:
        effectiveness = A_Bx * desired_Bx_sign
        # 只保留正贡献
        effectiveness = np.maximum(effectiveness, 0.0)
    else:
        # 无明确偏好，使用绝对值
        effectiveness = effectiveness_Bx
    
    # 归一化权重
    total_eff = np.sum(effectiveness)
    if total_eff < 1e-12:
        # 如果 B_x 贡献都很小，使用均匀分布
        coil_weights = np.ones(8) / 8.0
    else:
        coil_weights = effectiveness / total_eff
    
    # 分析结果
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
    """
    生成基于目标点的初始电流
    
    Args:
        p_des: 目标点位置 (3,) [m]
        actuation_table_path: 磁场插值表路径
        p0: 初始点位置 (3,) [m]，默认 [0, 0, -0.05]
        I_magnitude: 总电流幅度 [A]，默认 15.0
        I_max: 单个线圈最大电流 [A]，默认 50.0
        m_body: 磁矩向量 (3,)，默认 [0, 0, 0.005301]
        min_coil_fraction: 最小线圈电流占比，低于此值的线圈设为0
        top_k_coils: 只激活前 k 个最有效的线圈，None 表示使用所有线圈
    
    Returns:
        I_init: 初始电流 (8,) [A]
        info: 分析信息字典
    """
    # 默认值
    if p0 is None:
        p0 = np.array([0.0, 0.0, -0.05])
    if m_body is None:
        m_body = np.array([0.0, 0.0, 0.005301])
    
    p_des = np.asarray(p_des, dtype=np.float64).reshape(3,)
    p0 = np.asarray(p0, dtype=np.float64).reshape(3,)
    
    # 加载磁场插值表
    table = load_actuation_table(actuation_table_path, dtype=jnp.float64)
    
    # 在目标点计算局部磁场梯度矩阵
    A_at_target = interpolate_A_checked(table, p_des)  # (8, 8)
    
    # 计算目标方向
    target_dir = compute_target_direction(p_des, p0)
    
    # 分析磁场梯度矩阵
    coil_weights, effectiveness, analysis = analyze_actuation_matrix(
        A_at_target, target_dir, m_body
    )
    
    # 应用 top-k 筛选
    if top_k_coils is not None and top_k_coils < 8:
        top_k = int(top_k_coils)
        # 找出前 k 个最有效的线圈
        top_indices = np.argsort(effectiveness)[-top_k:]
        mask = np.zeros(8, dtype=bool)
        mask[top_indices] = True
        coil_weights = np.where(mask, coil_weights, 0.0)
        # 重新归一化
        total = np.sum(coil_weights)
        if total > 1e-12:
            coil_weights = coil_weights / total
    
    # 应用最小占比阈值
    coil_weights = np.where(coil_weights >= min_coil_fraction, coil_weights, 0.0)
    
    # 重新归一化
    total = np.sum(coil_weights)
    if total < 1e-12:
        # 如果所有权重都被过滤掉，使用最有效的单个线圈
        max_idx = np.argmax(effectiveness)
        coil_weights = np.zeros(8)
        coil_weights[max_idx] = 1.0
    else:
        coil_weights = coil_weights / total
    
    # 生成初始电流
    I_init = coil_weights * I_magnitude
    
    # 限制在 I_max 范围内
    I_init = np.clip(I_init, -I_max, I_max)
    
    # 构建信息字典
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
        description="生成基于目标点的初始电流 I_init 和 I_ref"
    )
    parser.add_argument(
        "--p_des",
        type=str,
        required=True,
        help="目标点位置，格式：x,y,z (m)，例如：0.058,0,-0.118",
    )
    parser.add_argument(
        "--p0",
        type=str,
        default="0,0,-0.05",
        help="初始点位置，格式：x,y,z (m)，默认：0,0,-0.05",
    )
    parser.add_argument(
        "--actuation_table",
        type=str,
        default="supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl",
        help="磁场插值表路径",
    )
    parser.add_argument(
        "--I_magnitude",
        type=float,
        default=15.0,
        help="总电流幅度 [A]，默认：15.0",
    )
    parser.add_argument(
        "--I_max",
        type=float,
        default=50.0,
        help="单个线圈最大电流 [A]，默认：50.0",
    )
    parser.add_argument(
        "--min_coil_fraction",
        type=float,
        default=0.05,
        help="最小线圈电流占比，低于此值的线圈设为0，默认：0.05",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="只激活前 k 个最有效的线圈，默认：None（使用所有线圈）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径，默认：打印到终端",
    )
    
    args = parser.parse_args()
    
    # 解析目标点
    p_des_vals = [float(x.strip()) for x in args.p_des.split(",")]
    if len(p_des_vals) != 3:
        raise ValueError(f"p_des 必须是3个值，得到：{len(p_des_vals)}")
    p_des = np.array(p_des_vals)
    
    # 解析初始点
    p0_vals = [float(x.strip()) for x in args.p0.split(",")]
    if len(p0_vals) != 3:
        raise ValueError(f"p0 必须是3个值，得到：{len(p0_vals)}")
    p0 = np.array(p0_vals)
    
    # 生成初始电流
    I_init, info = generate_current_init(
        p_des=p_des,
        actuation_table_path=args.actuation_table,
        p0=p0,
        I_magnitude=args.I_magnitude,
        I_max=args.I_max,
        min_coil_fraction=args.min_coil_fraction,
        top_k_coils=args.top_k,
    )
    
    # 输出结果
    output_data = {
        "I_init": I_init.tolist(),
        "I_ref": I_init.tolist(),  # I_ref 与 I_init 相同
        "info": info,
    }
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"结果已保存到：{output_path}")
    else:
        print(json.dumps(output_data, indent=2))
    
    # 打印摘要
    print("\n=== 初始电流生成摘要 ===")
    print(f"目标点: {p_des}")
    print(f"初始点: {p0}")
    print(f"I_init: {I_init}")
    print(f"激活线圈数: {info['active_coils']}")
    print(f"最大电流: {info['max_current']:.2f} A")
    print(f"线圈权重: {np.array(info['coil_weights'])}")
    
    # 打印各线圈的电流
    print("\n各线圈电流分配:")
    for i, (current, weight) in enumerate(zip(I_init, info['coil_weights'])):
        if abs(current) > 0.01:
            print(f"  Coil {i}: {current:6.2f} A (权重: {weight:.3f})")


if __name__ == "__main__":
    main()
