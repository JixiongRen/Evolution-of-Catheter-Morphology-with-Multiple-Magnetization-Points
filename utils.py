import numpy as np
from typing import Callable, Optional, Tuple, TYPE_CHECKING
from basic_utils import skew, quat_to_rotmat, quat_derivative, GL3_A, GL3_B, GL3_C
from segments import FlexibleSegment, RigidSegment

if TYPE_CHECKING:
    from catheter import RodMesh
    from equilibrium_solver import EquilibriumSolver

# -------- 工具函数 --------


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
        res_Q = Q - Q0_target   # 也可以只用单位四元数 + 方向约束的形式

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


def make_initial_guess(mesh: "RodMesh", rigid: RigidSegment):
    """
    根据 mesh (M 个区间) 构造一个简单的初始猜测:
    - 柔性段 M+1 个网格点沿 z 轴直线伸出
    - 刚性段继续沿 z 轴
    - 内力/内矩、k 都先设为 0
    """
    # 延迟导入以避免循环依赖
    from catheter import RodMesh
    from equilibrium_solver import EquilibriumSolver
    
    M = mesh.M
    Lf = mesh.flex_seg.length
    Lr = rigid.length

    # 近端位置在原点，沿 z 方向展开
    # 我们用 sigma_nodes / Lf 的比例决定每个节点的位置
    sigmas = mesh.sigma_nodes
    z_positions = sigmas / Lf * Lf  # 其实就是 sigmas，本例里就是 0..Lf

    # 统一的四元数
    Q0 = np.array([1.0, 0.0, 0.0, 0.0])

    x_nodes = np.zeros((M + 1, 13))
    for n in range(M + 1):
        p = np.array([0.0, 0.0, z_positions[n]])
        f = np.zeros(3)
        tau = np.zeros(3)
        x_nodes[n] = np.concatenate([p, Q0, f, tau])

    # 刚性段末端：在柔性段末端基础上再往 z 方向延 Lr
    p_flex_end = x_nodes[-1][:3]
    p_rigid_end = p_flex_end + np.array([0.0, 0.0, Lr])
    f_rigid = np.zeros(3)
    tau_rigid = np.zeros(3)
    x_rigid = np.concatenate([p_rigid_end, Q0, f_rigid, tau_rigid])

    # k_array: 全零
    k_array = np.zeros((M, 3, 13))

    # 打包 z0
    solver_dummy = EquilibriumSolver(
        mesh=mesh,
        rigid_seg=rigid,
        p0_target=np.array([0.0, 0.0, 0.0]),
        Q0_target=Q0,
        f_ext_rigid=np.zeros(3),
        tau_ext_rigid=np.zeros(3),
    )
    z0 = solver_dummy.pack_z(x_nodes, k_array, x_rigid)
    return z0, solver_dummy  # 返回一个临时 solver，只是为了用它的 pack_z
