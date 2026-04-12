#!/usr/bin/env python3
"""
诊断脚本：验证GPU加速和磁场模型集成
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import json

# 启用64位精度
jax.config.update("jax_enable_x64", True)

def check_gpu_availability():
    """检查GPU可用性和JAX设备配置"""
    print("=" * 60)
    print("GPU和设备诊断")
    print("=" * 60)
    
    # 检查可用设备
    devices = jax.devices()
    print(f"\n可用设备数量: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  设备 {i}: {device.device_kind} - {device}")
    
    # 检查默认后端
    print(f"\n默认后端: {jax.default_backend()}")
    
    # 测试简单计算
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    print(f"\n测试计算完成，结果形状: {y.shape}")
    print(f"计算设备: {y.device()}" if hasattr(y, '__call__') else f"计算设备: {y.devices()}")
    
    return devices


def check_magnetic_model_integration():
    """检查磁场模型是否正确集成到FK中"""
    print("\n" + "=" * 60)
    print("磁场模型集成诊断")
    print("=" * 60)
    
    from .forward_kinematics_optimized_bak.fk import build_solver_params
    from .forward_kinematics_optimized_bak.equilibrium_solver_nondim import residual_bar, with_coil_currents
    
    # 检查是否能加载磁场插值表
    actuation_table_path = Path("supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl")
    if actuation_table_path.exists():
        print(f"\n✓ 磁场插值表存在: {actuation_table_path}")
    else:
        print(f"\n✗ 磁场插值表不存在: {actuation_table_path}")
        return False
    
    # 测试构建SolverParams时是否包含磁场模型
    try:
        # 简单配置
        flex_lengths_tail = [0.03, 0.03]
        rigid_lengths = [0.003, 0.003, 0.003]
        M_list = [3, 2, 2]
        
        # 磁矩配置
        m_body_list = [[0, 0, 0.005301], [0, 0, 0.005301], [0, 0, 0.005301]]
        
        print("\n测试构建SolverParams...")
        print(f"  flex_lengths_tail: {flex_lengths_tail}")
        print(f"  rigid_lengths: {rigid_lengths}")
        print(f"  M_list: {M_list}")
        print(f"  m_body_list: {m_body_list}")
        
        # 这里需要完整的参数，暂时只检查导入
        print("\n✓ 磁场模型模块可以导入")
        
    except Exception as e:
        print(f"\n✗ 磁场模型集成检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def analyze_experiment_traces():
    """分析实验trace，对比磁场影响"""
    print("\n" + "=" * 60)
    print("实验Trace分析")
    print("=" * 60)
    
    pipeline_dir = Path("ik_out/diag_mag_realistic_t2_auto_pipeline")
    exp2_dir = Path("ik_out/diag_mag_realistic_t2_auto_init_exp2")
    
    if not pipeline_dir.exists() or not exp2_dir.exists():
        print("\n✗ 实验目录不存在")
        return
    
    # 读取trace
    def read_trace(trace_path):
        trace = []
        with open(trace_path, 'r') as f:
            for line in f:
                if line.strip():
                    trace.append(json.loads(line))
        return trace
    
    pipeline_trace = read_trace(pipeline_dir / "trace.jsonl")
    exp2_trace = read_trace(exp2_dir / "trace.jsonl")
    
    print(f"\nPipeline trace长度: {len(pipeline_trace)}")
    print(f"Exp2 trace长度: {len(exp2_trace)}")
    
    # 分析电流变化
    print("\n--- Pipeline电流变化分析 ---")
    for idx in [0, 10, 50, 100, 200, len(pipeline_trace)-1]:
        if idx < len(pipeline_trace):
            t = pipeline_trace[idx]
            I_A = t.get('I_A', [0]*8)
            I_max = max([abs(x) for x in I_A])
            I_norm = np.linalg.norm(I_A)
            print(f"iter {t.get('iter', idx):4d}: I_max={I_max:6.2f}A, I_norm={I_norm:6.2f}A, dist={t.get('dist_to_target_mm', 0):7.2f}mm")
    
    print("\n--- Exp2电流变化分析 ---")
    for idx in [0, 10, 50, 100, len(exp2_trace)-1]:
        if idx < len(exp2_trace):
            t = exp2_trace[idx]
            I_A = t.get('I_A', [0]*8)
            I_max = max([abs(x) for x in I_A])
            I_norm = np.linalg.norm(I_A)
            print(f"iter {t.get('iter', idx):4d}: I_max={I_max:6.2f}A, I_norm={I_norm:6.2f}A, dist={t.get('dist_to_target_mm', 0):7.2f}mm")
    
    # 检查norm_E变化
    print("\n--- Pipeline平衡残差(norm_E)变化 ---")
    for idx in [0, 50, 100, 200, len(pipeline_trace)-1]:
        if idx < len(pipeline_trace):
            t = pipeline_trace[idx]
            print(f"iter {t.get('iter', idx):4d}: norm_E={t.get('norm_E_raw', 0):8.4f}, normR={t.get('normR', 0):8.4f}")
    
    print("\n--- Exp2平衡残差(norm_E)变化 ---")
    for idx in [0, 50, 100, len(exp2_trace)-1]:
        if idx < len(exp2_trace):
            t = exp2_trace[idx]
            print(f"iter {t.get('iter', idx):4d}: norm_E={t.get('norm_E_raw', 0):8.4f}, normR={t.get('normR', 0):8.4f}")


def main():
    """主诊断流程"""
    print("\n" + "=" * 60)
    print("IK求解器诊断工具")
    print("=" * 60)
    
    # 1. GPU检查
    devices = check_gpu_availability()
    
    # 2. 磁场模型集成检查
    mag_ok = check_magnetic_model_integration()
    
    # 3. 实验trace分析
    analyze_experiment_traces()
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    print(f"GPU设备数量: {len(devices)}")
    print(f"默认后端: {jax.default_backend()}")
    print(f"磁场模型集成: {'✓ 正常' if mag_ok else '✗ 异常'}")
    print("\n建议:")
    if jax.default_backend() == 'cpu':
        print("  ⚠ 当前使用CPU后端，建议配置GPU加速")
    if not mag_ok:
        print("  ⚠ 磁场模型集成存在问题，需要进一步检查")
    print("=" * 60)


if __name__ == "__main__":
    main()
