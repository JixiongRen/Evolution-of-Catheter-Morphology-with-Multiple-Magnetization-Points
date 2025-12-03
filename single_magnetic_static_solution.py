"""
脚本名称：single_magnetic_static_solution.py

功能概述
---------
本脚本用于求解由“单个柔性段 + 单个刚性磁体段”组成的磁驱导管在静力学下的平衡配置，并以三维方式进行可视化。
求解器采用类 Levenberg–Marquardt（LM）方法对离散化后的 Cosserat 方程与边界条件进行最小二乘求解。

模型结构
---------
- 柔性段（FlexibleSegment）：采用 Cosserat 杆模型离散，使用 Gauss–Legendre（GL3）规则在每个小区间内进行积分与残差构造。
- 刚性段（RigidSegment）：作为一段长度固定、方向由柔性末端状态推进得到的末端段，可在其上施加外力 f_ext_rigid 与力矩 tau_ext_rigid。

边界条件
---------
- 近端（基座）姿态固定：位置 p0_target 与四元数姿态 Q0_target 作为硬约束。
- 远端（刚性段末端）自由：通过“零合力/零合矩”形式的边界条件函数施加（在 EquilibriumSolver 内部的 C_BV_dist）。

求解流程（demo_one_flex_one_rigid）
----------------------------------
1) 构建柔性段与刚性段对象，设置长度与刚度矩阵等参数；
2) 构建柔性段网格（RodMesh），例如将柔性段分成若干 GL3 区间；
3) 设定基座姿态边界条件（p0_target, Q0_target）及刚性段上的外载（f_ext_rigid, tau_ext_rigid）；
4) 组装初始猜测变量 z0（包含柔性段各节点状态、区间斜率和刚性段末端状态）；
5) 采用 LM 迭代，最小化所有区间的几何/本构残差与边界条件残差；
6) 将得到的平衡解转换为空间曲线并绘制三维形状。

可视化与扫描示例
-----------------
脚本内提供两个与可视化相关的函数：
- plot_catheter_3d：给定解，直接绘制单条导管的 3D 姿态；
- compute_catheter_points：仅计算柔性+刚性段的空间采样点，便于自定义绘图。

演示函数 demo_one_flex_one_rigid 还实现了一个“力矩方向扫描”的示例：
- 将刚性段所受外加力矩的模值固定为 |tau|=15；
- 在 xOy 平面内均匀采样 16 个方向做一周旋转；
- 对每个方向分别求解静力平衡（用上一方向的解作为热启动以加速收敛）；
- 将 16 条导管姿态绘制在同一幅 3D 图中，并用虚线连接末端点以给出末端轨迹。

使用方法
--------
直接运行本脚本：
    python single_magnetic_static_solution.py
运行后会依次完成 16 个力矩方向的求解，并弹出一张 3D 图显示所有姿态与末端轨迹。

可调参数
--------
- 柔性段/刚性段长度与刚度矩阵（FlexibleSegment / RigidSegment 初始化参数）；
- 柔性段网格密度（RodMesh 的 n_intervals 或自定义 sigma_nodes）；
- 力矩扫描的幅值与方向数（tau_mag, n_dirs）；
- 刚性段采样点数（compute_catheter_points 的 n_samples_rigid）。

依赖
----
numpy、matplotlib 以及本项目中的 basic_utils、segments、equilibrium_solver、catheter 等模块。
"""

import numpy as np
import matplotlib.pyplot as plt

from basic_utils import GL3_A, GL3_B, GL3_C
from segments import FlexibleSegment, RigidSegment
from equilibrium_solver import EquilibriumSolver
from catheter import RodMesh


def compute_catheter_points(
        mesh: RodMesh,
        rigid: RigidSegment,
        x_nodes_star: np.ndarray,
        f_ext_rigid: np.ndarray,
        tau_ext_rigid: np.ndarray,
        n_samples_rigid: int = 20,
) -> np.ndarray:
    """
    仅计算整根导管在 3D 空间中的采样点 P（柔性段 + 刚性段），不绘图。
    返回 P，shape = (M+1 + n_samples_rigid, 3)
    """
    # 柔性段所有网格点的位置
    p_flex = x_nodes_star[:, :3]

    # 刚性段沿长度方向采样
    x_flex_end = x_nodes_star[-1]
    Lr = rigid.length
    sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)

    p_rigid_list = []
    for s in sigmas_rigid[1:]:
        x_s = rigid.state_along(
            x_proximal=x_flex_end,
            sigma=s,
            f_ext_total=f_ext_rigid,
            tau_ext_total=tau_ext_rigid,
        )
        p_rigid_list.append(x_s[:3])

    p_rigid = np.array(p_rigid_list)
    P = np.vstack([p_flex, p_rigid])
    return P


def demo_one_flex_one_rigid():
    # 1. 创建一段柔性段
    flex = FlexibleSegment(
        length=0.1,
        K_se=np.eye(3),
        K_bt=np.eye(3),
    )

    # 2. 创建一段刚性段
    rigid = RigidSegment(
        length=0.02,
        v_star=np.array([0.0, 0.0, 1.0]),
    )

    # 3. 网格: 例如把柔性段分成 5 个 Gauss–Legendre 区间
    mesh = RodMesh(flex_seg=flex, n_intervals=5)
    # 或者自定义 sigma_nodes：
    # mesh = RodMesh(flex_seg=flex, sigma_nodes=np.array([0.0, 0.02, 0.05, 0.1]))

    # 4. 边界条件: 近端 pose 固定, 远端 wrench=0
    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    f_ext_rigid = np.array([0., 0., 0.])
    tau_ext_rigid = np.array([0., 0., 0.])

    # 5. 构造 solver
    solver = EquilibriumSolver(
        mesh=mesh,
        rigid_seg=rigid,
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_rigid=f_ext_rigid,
        tau_ext_rigid=tau_ext_rigid,
        max_iter=20000,
        tol=1e-5,
        lm_damping=1e-3,
    )

    # 6. 构造初始猜测
    M = mesh.M
    sigmas = mesh.sigma_nodes
    # 柔性段节点
    x_nodes = np.zeros((M + 1, 13))
    for n in range(M + 1):
        p = np.array([0.0, 0.0, sigmas[n]])
        Q = Q0_target
        f = np.zeros(3)
        tau = np.zeros(3)
        x_nodes[n] = np.concatenate([p, Q, f, tau])

    # 刚性段末端
    p_flex_end = x_nodes[-1][:3]
    p_rigid_end = p_flex_end + np.array([0.0, 0.0, rigid.length])
    x_rigid = np.concatenate([p_rigid_end, Q0_target, np.zeros(3), np.zeros(3)])

    # k_array
    k_array = np.zeros((M, 3, 13))

    z0 = solver.pack_z(x_nodes, k_array, x_rigid)
    
    # 7. 在 xOy 面内旋转一周的力矩，模值为 15，均匀 16 个方向
    n_dirs = 16
    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    tau_mag = 15.0

    # 统一绘制在同一个 3D 图中
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.hsv(np.linspace(0, 1, n_dirs, endpoint=False))

    end_points = []
    z_init = z0.copy()

    for i, theta in enumerate(angles):
        tau_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
        tau_i = tau_mag * tau_dir
        # 更新求解器中的刚性段外加力矩
        solver.tau_ext_rigid = tau_i
        solver.f_ext_rigid = f_ext_rigid

        # 求解（使用上一次解作为初值以加速收敛）
        z_star, success = solver.solve(z_init)
        print(f"LM success (i={i}, theta={theta:.3f}):", success)
        x_nodes_star, k_array_star, x_rigid_star = solver.unpack_z(z_star)

        # 计算整根导管的采样点并绘制
        P = compute_catheter_points(
            mesh=mesh,
            rigid=rigid,
            x_nodes_star=x_nodes_star,
            f_ext_rigid=f_ext_rigid,
            tau_ext_rigid=tau_i,
            n_samples_rigid=20,
        )
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=colors[i], marker='o', markersize=3, linewidth=1.5)

        # 记录末端点
        end_points.append(P[-1])

        # 下一个方向用当前解热启动
        z_init = z_star

    # 用虚线连接末端轨迹（闭合）
    end_points = np.array(end_points)
    end_points_closed = np.vstack([end_points, end_points[0]])
    ax.plot(end_points_closed[:, 0], end_points_closed[:, 1], end_points_closed[:, 2], 'k--', linewidth=1.5, label='end-tip trajectory')

    # 统一设置坐标系外观
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('16 catheter poses with rotating torque in xOy (|tau|=15)')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([0, 0.15])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_one_flex_one_rigid()