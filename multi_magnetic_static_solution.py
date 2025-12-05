# multi_segment_equilibrium_solver.py

from __future__ import annotations
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from module.equilibrium_solver import MultiSegmentEquilibriumSolver
from module.segments import RigidSegment
from module.catheter import RodMesh
from module.utils import (
    make_initial_guess_multi,
)


# ------------------- 可视化：多段导管最终姿态 -------------------
def plot_catheter_3d_multiseg(
    meshes: List[RodMesh],
    rigid_segs: List[RigidSegment],
    x_nodes_list_star: List[np.ndarray],
    x_rigid_list_star: List[np.ndarray],
    f_ext_list: List[np.ndarray],
    tau_ext_list: List[np.ndarray],
    n_samples_rigid: int = 10,
):
    """
    基于最终解（x_nodes_list_star, x_rigid_list_star），拼接 N 对（柔性+刚性）段的空间点并绘制 3D 曲线。
    刚性段通过 RigidSegment.state_along 从对应柔性末端推进，使用各段的 f_ext_list/tau_ext_list。
    """
    N = len(rigid_segs)
    P_all = []  # 收集所有段的点，依次拼接

    for i in range(N):
        mesh_i = meshes[i]
        rigid_i = rigid_segs[i]
        x_nodes_i = x_nodes_list_star[i]  # (Mi+1, 13)

        # 柔性段所有网格点位置
        p_flex_i = x_nodes_i[:, :3]
        # 避免重复连接点：除第一段外，去掉首个点
        if i > 0:
            p_flex_i = p_flex_i[1:, :]
        P_all.append(p_flex_i)

        # 刚性段：从柔性末端出发沿长度采样
        x_flex_end_i = x_nodes_i[-1]
        Lr = rigid_i.length
        sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)

        p_rigid_i = []
        for s in sigmas_rigid[1:]:  # 跳过 0
            x_s = rigid_i.state_along(
                x_proximal=x_flex_end_i,
                sigma=s,
                f_ext_total=f_ext_list[i],
                tau_ext_total=tau_ext_list[i],
            )
            p_rigid_i.append(x_s[:3])
        p_rigid_i = np.array(p_rigid_i)
        P_all.append(p_rigid_i)

    P = np.vstack(P_all)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:, 0], P[:, 1], P[:, 2], marker='o', markersize=3, linewidth=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multi-segment catheter final configuration')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([0, 0.25])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) 构建 3 段柔性 + 3 段刚性
    from module.segments import FlexibleSegment, RigidSegment
    from module.catheter import RodMesh

    # 你可以根据论文设定更真实的长度/刚度
    flex1 = FlexibleSegment(length=0.06, K_se=np.eye(3), K_bt=np.eye(3))
    flex2 = FlexibleSegment(length=0.06, K_se=np.eye(3), K_bt=np.eye(3))
    flex3 = FlexibleSegment(length=0.06, K_se=np.eye(3), K_bt=np.eye(3))

    # 每段柔性都各自有一个 RodMesh（GL3 区间数可以各不相同）
    mesh1 = RodMesh(flex_seg=flex1, n_intervals=4)
    mesh2 = RodMesh(flex_seg=flex2, n_intervals=4)
    mesh3 = RodMesh(flex_seg=flex3, n_intervals=4)

    rigid1 = RigidSegment(length=0.01, v_star=np.array([0., 0., 1.]))
    rigid2 = RigidSegment(length=0.01, v_star=np.array([0., 0., 1.]))
    rigid3 = RigidSegment(length=0.01, v_star=np.array([0., 0., 1.]))

    flex_segs = [flex1, flex2, flex3]
    meshes = [mesh1, mesh2, mesh3]
    rigid_segs = [rigid1, rigid2, rigid3]

    # 2) 边界条件：近端 pose 固定
    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    # 3) 每段刚体上的外载（这里随便设一个例子：只在每段上给一点扭矩）
    f_ext_list = [
        np.array([3.4, -3.7, 0.4]) * 1e-3,
        np.array([-8.5, 2.1, 3.3]) * 1e-3,
        np.array([6.0, -1.3, -2.1]) * 1e-3,
    ]
    tau_ext_list = [
        np.array([-18.9, 4.7, 13.4]),
        np.array([-1.3, -19.1, 8.9]),
        np.array([19.4, 0.7, 12.6]),
    ]

    # 4) 初始猜测
    z0, x_nodes_list0, k_array_list0, x_rigid_list0 = make_initial_guess_multi(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
    )

    # 5) 构造多段求解器并求解
    solver = MultiSegmentEquilibriumSolver(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_list=f_ext_list,
        tau_ext_list=tau_ext_list,
        max_iter=200000,
        tol=1e-2,
        lm_damping=1e-3,
    )

    z_star, success = solver.solve(z0)
    print("Multi-segment LM success:", success)

    x_nodes_list_star, k_array_list_star, x_rigid_list_star = solver.unpack_z(z_star)

    # 简单打印一下各段末端的位置
    for i in range(3):
        p_flex_end = x_nodes_list_star[i][-1, :3]
        p_rigid_end = x_rigid_list_star[i][:3]
        print(f"Segment pair {i+1}:")
        print("  F end p =", p_flex_end)
        print("  R end p =", p_rigid_end)

    # 成功则绘制最终的多段导管姿态
    if success:
        plot_catheter_3d_multiseg(
            meshes=meshes,
            rigid_segs=rigid_segs,
            x_nodes_list_star=x_nodes_list_star,
            x_rigid_list_star=x_rigid_list_star,
            f_ext_list=f_ext_list,
            tau_ext_list=tau_ext_list,
            n_samples_rigid=20,
        )
