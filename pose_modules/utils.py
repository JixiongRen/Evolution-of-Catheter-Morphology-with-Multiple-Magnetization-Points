import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, TYPE_CHECKING, List
from .basics import skew, quat_to_rotmat, quat_derivative, GL3_A, GL3_B, GL3_C
from .rod_mesh import RodMesh
from .segments import FlexibleSegment, RigidSegment
from .external_wrench import GravityLineDensity, GravityRigid


# -------- LM算法工具函数 --------


def make_C_S_flexible(env_constraint: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """
    返回一个柔性段专用的 C_S_fun 状态约束函数
    - 对 x_n 和 x_{n+1} 的四元数施加单位长度约束
    - 可选: 对位置等施加环境约束
    """

    def C_S_fun(x_n: np.ndarray, x_np1: np.ndarray) -> np.ndarray:
        Qn = x_n[3:7]
        Qnp1 = x_np1[3:7]

        # 四元数单位约束
        quat_res_n = np.array([np.dot(Qn, Qn) - 1.])
        quat_res_np1 = np.array([np.dot(Qnp1, Qnp1) - 1.])

        res_list = [quat_res_n, quat_res_np1]

        if env_constraint is not None:
            res_list.append(env_constraint(x_n))
            res_list.append(env_constraint(x_np1))

        return np.concatenate(res_list)

    return C_S_fun


def make_C_S_rigid(
        rigid_seg: RigidSegment,
        f_ext_total: np.ndarray,
        tau_ext_total: np.ndarray,
        env_constraint: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """
    返回给刚性段用的 C_S_fun(x_n, x_np1)
    - 里面调用 rigid_seg.propagate(x_n, f_ext_total, tau_ext_total) 得到解析 x_rigid(L)
    - 施加 x_{n+1} - x_rigid(L) = 0
    - 施加单位四元数约束
    """

    def C_S_fun(x_n: np.ndarray, x_np1: np.ndarray) -> np.ndarray:
        x_rigid_end = rigid_seg.propagate(x_n, f_ext_total, tau_ext_total)

        core = x_np1 - x_rigid_end
        Q_np1 = x_np1[3:7]
        quat_res = np.array([np.dot(Q_np1, Q_np1) - 1.0])

        res_list = [core, quat_res]

        if env_constraint is not None:
            res_list.append(env_constraint(x_n))
            res_list.append(env_constraint(x_np1))

        return np.concatenate(res_list)

    return C_S_fun


def make_C_BV_proximal_pose(p0_target: np.ndarray, Q0_target: np.ndarray):
    """
    固定近端位置和姿态:
    p(0) = p0_target, Q(0) = Q0_target
    """

    def C_BV_fun(x_n: np.ndarray, x_np1: np.ndarray) -> np.ndarray:
        # 这个函数会在包含 proximal 状态的那个小区间里被调用
        # 约束 x_n
        p = x_n[:3]
        Q = x_n[3:7]

        res_p = p - p0_target
        res_Q = Q - Q0_target  # 也可以只用单位四元数 + 方向约束的形式

        return np.concatenate([res_p, res_Q])

    return C_BV_fun


def make_C_BV_distal_free_tip():
    """
    distal 端零 wrench 条件:
    f(L) = 0, tau(L) = 0
    """

    def C_BV_fun(x_n: np.ndarray, x_np1: np.ndarray) -> np.ndarray:
        # 在 distal 端所在的区间上调用: 约束 x_{n+1} 的 wrench
        f = x_np1[7:10]
        tau = x_np1[10:13]
        return np.concatenate([f, tau])

    return C_BV_fun


def make_initial_guess_multi(
        flex_segs: List[FlexibleSegment],
        meshes: List[RodMesh],
        rigid_segs: List[RigidSegment],
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    构造 3F+3R（或通用 N 对）的初始猜测:
    - 整根沿 z 轴直线展开
    - 每段柔性在自己的 sigma_nodes 上均匀拉直
    - 每个刚性段在前一段末端基础上增加其 length
    - 内力/内矩和所有 k 全部设为 0
    返回：
    - z0
    - x_nodes_list, k_array_list, x_rigid_list （方便你单独用来做可视化等）
    """
    from .rod_mesh import RodMesh
    from .equilibrium_solver import MultiSegmentEquilibriumSolver

    N = len(flex_segs)
    assert len(flex_segs) == len(meshes) == len(rigid_segs)

    x_nodes_list: List[np.ndarray] = []
    k_array_list: List[np.ndarray] = []
    x_rigid_list: List[np.ndarray] = []

    z_base = 0.0  # 全局 z 坐标累计
    Q0 = np.array([1.0, 0.0, 0.0, 0.0])

    # 1) 柔性 + 刚性依次排布
    for i in range(N):
        flex = flex_segs[i]
        mesh = meshes[i]
        rigid = rigid_segs[i]

        M = mesh.M
        sigmas = mesh.sigma_nodes

        # 柔性段节点
        x_nodes = np.zeros((M + 1, 13))
        for n in range(M + 1):
            p = np.array([0.0, 0.0, z_base + sigmas[n]])
            f = np.zeros(3)
            tau = np.zeros(3)
            x_nodes[n] = np.concatenate([p, Q0, f, tau])
        x_nodes_list.append(x_nodes)

        # 柔性段结束后，更新 z_base
        z_base += flex.length

        # 刚性段末端
        p_rigid_end = np.array([0.0, 0.0, z_base + rigid.length])
        fR = np.zeros(3)
        tauR = np.zeros(3)
        xR = np.concatenate([p_rigid_end, Q0, fR, tauR])
        x_rigid_list.append(xR)

        # 刚性段结束后，继续更新 z_base
        z_base += rigid.length

        # k_array 初始为 0
        k_array = np.zeros((M, 3, 13))
        k_array_list.append(k_array)

    # 2) 打包成 z0
    dummy_solver = MultiSegmentEquilibriumSolver(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        p0_target=np.array([0.0, 0.0, 0.0]),
        Q0_target=Q0,
        f_ext_list=[np.zeros(3) for _ in range(N)],
        tau_ext_list=[np.zeros(3) for _ in range(N)],
    )
    z0 = dummy_solver.pack_z(x_nodes_list, k_array_list, x_rigid_list)
    return z0, x_nodes_list, k_array_list, x_rigid_list


# ------------------- 材料工具函数 -------------------


# def build_k_matrices_for_pdms(d_outer: float) -> tuple[np.ndarray, np.ndarray]:
#     """
#     根据 PDMS 材料和圆截面直径，构造柔性段的 K_se, K_bt.
#     假定:
#       - E = 1.8 MPa
#       - G = 0.6 MPa
#       - 截面直径 d_outer (m)
#
#     K_se = diag(GA, EA, EA)
#     K_bt = diag(EI_xx, EI_yy, EI_zz)
#     """
#     E = 0.9e6      # Pa
#     G = 0.3e6      # Pa
#     r = d_outer / 2.0
#     A = np.pi * r**2
#
#     # 面积矩: 圆截面
#     Ixx = Iyy = np.pi * r**4 / 4.0
#     Izz = np.pi * r**4 / 2.0   # 极惯性矩
#
#     K_se = np.diag([G * A, E * A, E * A])
#     K_bt = np.diag([E * Ixx, E * Iyy, E * Izz])
#
#     return K_se, K_bt


def build_k_matrices_for_pdms(d_outer: float):
    """
    PDMS flexible segment stiffness matrices (Cosserat rod).

    Parameters
    ----------
    d_outer : float
        Outer diameter [m]

    Assumptions
    -----------
    E = 1.8 MPa
    G = 0.6 MPa
    Circular cross-section
    """

    E = 1.8e6  # Pa
    G = 0.6e6  # Pa

    r = d_outer / 2.0
    A = np.pi * r**2

    Ixx = Iyy = np.pi * r**4 / 4.0
    J   = np.pi * r**4 / 2.0

    K_se = np.diag([G * A, G * A, E * A])
    K_bt = np.diag([E * Ixx, E * Iyy, G * J])

    return K_se, K_bt



def build_gravity_line_density_for_pdms(d_outer: float) -> GravityLineDensity:
    """
    PDMS 柔性段重力线密度：
      - ρ = 970 kg/m^3
      - A = π r^2
      - line_mass = ρ A
      - line_force = line_mass * g
      - g_vec = [0, 0, -1] （方向），rhoA 里已经乘 9.81
    """
    rho = 970.0  # kg/m^3
    r = d_outer / 2.0
    A = np.pi * r**2
    line_mass = rho * A         # kg/m
    rhoA = line_mass * 9.81     # N/m
    g_vec = np.array([0.0, 0.0, -1.0])

    return GravityLineDensity(rhoA=rhoA, g_vec=g_vec)


def build_gravity_rigid_for_ndfeb(d_outer: float, length: float) -> GravityRigid:
    """
    NdFeB 刚体段重力：
      - ρ = 7500 kg/m^3
      - A = π r^2
      - V = A * L
      - mass = ρ V
      - g_world = [0, 0, -9.81]
      - r_cm_body = [0, 0, L/2]
    """
    rho = 7500.0  # kg/m^3
    r = d_outer / 2.0
    A = np.pi * r**2
    V = A * length
    mass = rho * V
    g_world = np.array([0.0, 0.0, -9.81])
    r_cm_body = np.array([0.0, 0.0, length / 2.0])

    return GravityRigid(mass=mass, g_world=g_world, r_cm_body=r_cm_body)


# ------------------- 绘制 -------------------

def plot_catheter_3d(
        mesh: RodMesh,
        rigid: RigidSegment,
        x_nodes_star: np.ndarray,  # (M+1, 13) 柔性段网格点解
        x_rigid_star: np.ndarray,  # (13,) 刚性段末端解
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
    p_flex = x_nodes_star[:, :3]  # shape (M+1, 3)

    # 2. 刚性段采样若干点
    x_flex_end = x_nodes_star[-1]  # 刚性段起点状态
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

    p_rigid = np.array(p_rigid_list)  # shape (n_samples_rigid, 3)

    # 3. 拼接整根导管的所有点
    #    注意顺序：柔性段所有点 + 刚性段采样点
    P = np.vstack([p_flex, p_rigid])  # shape (M+1 + n_samples_rigid, 3)

    # 4. 三维绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(P[:, 0], P[:, 1], P[:, 2], marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Catheter configuration')

    ax.view_init(elev=30, azim=-60)  # 可以随时调整观察角度
    ax.set_box_aspect([1, 1, 1])  # 使 xyz 比例一致

    # 设置x轴范围
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([0, 0.15])

    plt.tight_layout()
    plt.show()
