#!/usr/bin/env python3
"""
测试磁矩配置对IK求解的影响
运行相同目标点，改变m_body_list，观察最终姿态差异
"""
from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path
import numpy as np

def run_ik_experiment(exp_name: str, m_body_list: str, p_des: str = "0.068,-0.031,0.042"):
    """运行单个IK实验"""
    
    out_dir = f"ik_out/mag_influence_test_{exp_name}"
    
    cmd = [
        sys.executable,
        "-m",
        "single_layer_ik.auto_run_joint_ik",
        "--p_des", p_des,
        "--m_body_list", m_body_list,
        "--enable_magnetics",
        "--M_list", "3,2,2",  # 使用较低离散化以加快速度
        "--max_iter", "100",  # 限制迭代次数
        "--dist_tol_mm", "1.0",
        "--out_dir", out_dir,
    ]
    
    print(f"\n{'='*60}")
    print(f"运行实验: {exp_name}")
    print(f"m_body_list: {m_body_list}")
    print(f"输出目录: {out_dir}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return out_dir
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def compare_results(exp_dirs: dict):
    """比较不同实验的结果"""
    print(f"\n{'='*60}")
    print("实验结果对比")
    print(f"{'='*60}\n")
    
    results = {}
    for name, dir_path in exp_dirs.items():
        if dir_path is None:
            continue
        
        result_file = Path(dir_path) / "result.json"
        if not result_file.exists():
            print(f"✗ {name}: 结果文件不存在")
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
        print(f"  尖端位置: {results[name]['p_tip']}")
        print(f"  位置误差: {results[name]['pos_err_mm']:.4f} mm")
        print(f"  平衡残差: {results[name]['norm_E']:.6f}")
        print(f"  迭代次数: {results[name]['n_iter']}")
        print(f"  插入深度: {results[name]['x_m']:.6f} m")
        print(f"  电流: {[f'{x:.2f}' for x in results[name]['I_A']]}")
        print()
    
    # 计算姿态差异
    if len(results) >= 2:
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name1, name2 = names[i], names[j]
                p1 = np.array(results[name1]['p_tip'])
                p2 = np.array(results[name2]['p_tip'])
                diff = np.linalg.norm(p1 - p2) * 1000  # 转换为mm
                
                print(f"{name1} vs {name2}:")
                print(f"  尖端位置差异: {diff:.4f} mm")
                
                if diff < 1.0:
                    print(f"  ⚠ 警告: 差异小于1mm，磁矩配置可能未生效！")
                else:
                    print(f"  ✓ 差异显著，磁矩配置有影响")
                print()


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("磁场影响测试")
    print("="*60)
    
    # 定义测试配置
    # 测试1: 所有磁矩沿z轴正向
    # 测试2: 所有磁矩沿z轴负向
    # 测试3: 所有磁矩沿y轴正向
    
    experiments = {
        "z_positive": "0,0,0.005301; 0,0,0.005301; 0,0,0.005301",
        "z_negative": "0,0,-0.005301; 0,0,-0.005301; 0,0,-0.005301",
        "y_positive": "0,0.005301,0; 0,0.005301,0; 0,0.005301,0",
    }
    
    exp_dirs = {}
    for name, m_body in experiments.items():
        exp_dirs[name] = run_ik_experiment(name, m_body)
    
    # 对比结果
    compare_results(exp_dirs)
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
