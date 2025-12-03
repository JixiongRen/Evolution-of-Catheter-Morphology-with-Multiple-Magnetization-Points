import numpy as np
import matplotlib.pyplot as plt

from basic_utils import GL3_A, GL3_B, GL3_C
from segments import FlexibleSegment, RigidSegment
from equilibrium_solver import EquilibriumSolver
from catheter import RodMesh

def plot_catheter_3d(
        mesh: RodMesh,
        rigid: RigidSegment,
        x_nodes_star: np.ndarray,   # (M+1, 13) 柔性段网格点解
        x_rigid_star: np.ndarray,   # (13,) 刚性段末端解
        f_ext_rigid: np.ndarray,
        tau_ext_rigid: np.ndarray,
        n_samples_rigid: int = 10,
):
    """
    在三维空间中绘制整根导管的姿态曲线:
    - 柔性段：直接用 x_nodes_star 中的所有 p
    - 刚性段：从柔性段末端状态出发，用 state_along 采样若干点
    """

    # 1. 柔性段所有网格点的位置
    p_flex = x_nodes_star[:, :3]        # shape (M+1, 3)

    # 2. 刚性段采样若干点
    x_flex_end = x_nodes_star[-1]       # 刚性段起点状态
    Lr = rigid.length
    sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)  # 包含起点和末端

    p_rigid_list = []
    for s in sigmas_rigid[1:]:  # 跳过 s=0（已经是柔性末端）
        x_s = rigid.state_along(
            x_proximal=x_flex_end,
            sigma=s,
            f_ext_total=f_ext_rigid,
            tau_ext_total=tau_ext_rigid,
        )
        p_rigid_list.append(x_s[:3])

    p_rigid = np.array(p_rigid_list)    # shape (n_samples_rigid, 3)

    # 3. 拼接整根导管的所有点
    #    注意顺序：柔性段所有点 + 刚性段采样点
    P = np.vstack([p_flex, p_rigid])    # shape (M+1 + n_samples_rigid, 3)

    # 4. 三维绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(P[:, 0], P[:, 1], P[:, 2], marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Catheter configuration')

    ax.view_init(elev=30, azim=-60)  # 可以随时调整观察角度
    ax.set_box_aspect([1, 1, 1])     # 使 xyz 比例一致

    # 设置x轴范围
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([0, 0.15])

    plt.tight_layout()
    plt.show()


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

    f_ext_rigid = np.array([1., 0., 0.])
    tau_ext_rigid = np.array([0., 0., 0.])

    # 5. 构造 solver
    solver = EquilibriumSolver(
        mesh=mesh,
        rigid_seg=rigid,
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_rigid=f_ext_rigid,
        tau_ext_rigid=tau_ext_rigid,
        max_iter=20,
        tol=1e-8,
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

    # 7. 求解
    z_star, success = solver.solve(z0)

    print("LM success:", success)
    x_nodes_star, k_array_star, x_rigid_star = solver.unpack_z(z_star)
    print("x0* =", x_nodes_star[0])
    print("x_flex_end* =", x_nodes_star[-1])
    print("x_rigid_end* =", x_rigid_star)


    # 8. 绘制 3D 姿态曲线
    if success:
        plot_catheter_3d(
            mesh=mesh,
            rigid=rigid,
            x_nodes_star=x_nodes_star,
            x_rigid_star=x_rigid_star,
            f_ext_rigid=f_ext_rigid,
            tau_ext_rigid=tau_ext_rigid,
            n_samples_rigid=3,   # 刚性段上采样 20 个点，自己调
        )


if __name__ == "__main__":
    demo_one_flex_one_rigid()