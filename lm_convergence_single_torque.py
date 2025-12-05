"""
脚本名称：lm_convergence_single_torque.py

功能
----
可视化单一外加力矩下，LM 迭代过程中导管形态的收敛过程。
通过自实现一个与 EquilibriumSolver.solve 一致的 LM 循环，
在每次迭代后记录当前解并绘制对应的 3D 姿态曲线，
用由浅到深（透明度递增）的曲线展示从初始猜测到收敛解的变化。

说明
----
- 不修改现有模块，仅在本脚本内实现迭代可视化；
- 力矩默认设置为 |tau|=15，方向沿 x 轴；
- 仅绘制一个力矩情形；
- 为避免曲线过多，默认从所有迭代历史中等间隔抽取若干条进行绘制。

运行
----
python lm_convergence_single_torque.py

可调参数
--------
- tau_vec: 外加力矩向量（默认 [15, 0, 0]）
- max_iter, tol, lm_damping: LM 迭代参数
- n_intervals: 柔性段网格区间数
- n_samples_rigid: 刚性段沿长度方向采样点数（绘图用）
- n_snapshots: 从迭代历史中抽取的曲线数量上限
"""

import numpy as np
import matplotlib.pyplot as plt

from module.segments import FlexibleSegment, RigidSegment
from module.equilibrium_solver import SingleSegmentEquilibriumSolver
from module.catheter import RodMesh


def compute_catheter_points(
    mesh: RodMesh,
    rigid: RigidSegment,
    x_nodes: np.ndarray,
    f_ext_rigid: np.ndarray,
    tau_ext_rigid: np.ndarray,
    n_samples_rigid: int = 20,
) -> np.ndarray:
    """给定当前柔性段节点解 x_nodes，拼接刚性段采样点，返回整根导管的 3D 采样点 P。
    P 形状为 (M+1 + n_samples_rigid, 3)。"""
    # 柔性段所有网格点位置
    p_flex = x_nodes[:, :3]

    # 刚性段沿长度方向采样
    x_flex_end = x_nodes[-1]
    Lr = rigid.length
    sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)

    p_rigid_list = []
    for s in sigmas_rigid[1:]:  # 跳过 s=0（重合柔性末端）
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


def lm_convergence_demo():
    # 1) 构建段与网格
    flex = FlexibleSegment(
        length=0.1,
        K_se=np.eye(3),
        K_bt=np.eye(3),
    )
    rigid = RigidSegment(
        length=0.02,
        v_star=np.array([0.0, 0.0, 1.0]),
    )
    mesh = RodMesh(flex_seg=flex, n_intervals=5)

    # 2) 边界条件（近端固定姿态）
    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    # 3) 外载（仅考虑力矩），单一力矩情形
    f_ext_rigid = np.array([0.0, 0.0, 0.0])
    tau_vec = np.array([0., 15., 0.0])  # 可修改力矩方向/大小

    # 4) 构造求解器（仅用其中的残差/雅可比函数）
    solver = SingleSegmentEquilibriumSolver(
        mesh=mesh,
        rigid_seg=rigid,
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_rigid=f_ext_rigid,
        tau_ext_rigid=tau_vec,
        max_iter=20000,
        tol=1e-5,
        lm_damping=1e-3,
    )

    # 5) 初始猜测 z0
    M = mesh.M
    sigmas = mesh.sigma_nodes

    x_nodes0 = np.zeros((M + 1, 13))
    for n in range(M + 1):
        p = np.array([0.0, 0.0, sigmas[n]])
        Q = Q0_target
        f = np.zeros(3)
        tau = np.zeros(3)
        x_nodes0[n] = np.concatenate([p, Q, f, tau])

    p_flex_end = x_nodes0[-1][:3]
    p_rigid_end = p_flex_end + np.array([0.0, 0.0, rigid.length])
    x_rigid0 = np.concatenate([p_rigid_end, Q0_target, np.zeros(3), np.zeros(3)])

    k_array0 = np.zeros((M, 3, 13))
    z = solver.pack_z(x_nodes0, k_array0, x_rigid0)

    # 6) LM 迭代（自实现循环以记录中间状态）
    max_iter = 200  # 可调
    tol = 1e-5
    lm_damping = 1e-3

    z_history = []
    E_history = []

    for it in range(max_iter):
        E = solver.residual(z)
        normE = np.linalg.norm(E)
        z_history.append(z.copy())
        E_history.append(normE)
        print(f"[LM-visual] iter={it}, ||E||={normE:.3e}")
        if normE < tol:
            break

        J = solver.jacobian_fd(z, E)
        A = J.T @ J + lm_damping * np.eye(J.shape[1])
        g = J.T @ E
        delta = np.linalg.solve(A, g)
        z = z - delta

    # 7) 从迭代历史中抽取若干快照绘图
    n_snapshots = 12  # 最多绘制 12 条曲线
    idx_all = np.arange(len(z_history))
    if len(idx_all) <= n_snapshots:
        idx_pick = idx_all
    else:
        idx_pick = np.linspace(0, len(z_history) - 1, n_snapshots, dtype=int)

    # 透明度设置：从浅到深
    alphas = np.linspace(0.15, 1.0, len(idx_pick))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for a, idx in zip(alphas, idx_pick):
        x_nodes_i, k_array_i, x_rigid_i = solver.unpack_z(z_history[idx])
        P = compute_catheter_points(
            mesh=mesh,
            rigid=rigid,
            x_nodes=x_nodes_i,
            f_ext_rigid=f_ext_rigid,
            tau_ext_rigid=tau_vec,
            n_samples_rigid=20,
        )
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=(0.2, 0.4, 0.8, a), linewidth=2)

    # 标注初末状态
    x_nodes_first, _, _ = solver.unpack_z(z_history[idx_pick[0]])
    x_nodes_last, _, _ = solver.unpack_z(z_history[idx_pick[-1]])
    P_first = compute_catheter_points(mesh, rigid, x_nodes_first, f_ext_rigid, tau_vec, 20)
    P_last = compute_catheter_points(mesh, rigid, x_nodes_last, f_ext_rigid, tau_vec, 20)
    ax.scatter(P_first[-1, 0], P_first[-1, 1], P_first[-1, 2], c='red', s=40, label='start tip')
    ax.scatter(P_last[-1, 0], P_last[-1, 1], P_last[-1, 2], c='green', s=40, label='final tip')

    ax.set_title('LM convergence under a single torque (|tau|=15, dir=x)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([0, 0.15])
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    lm_convergence_demo()
