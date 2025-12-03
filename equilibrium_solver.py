import numpy as np
from typing import Tuple
from segments import RigidSegment, FlexibleSegment
from catheter import RodMesh
from utils import make_C_S_flexible, make_C_BV_proximal_pose, make_C_BV_distal_free_tip

class EquilibriumSolver:
    def __init__(
            self,
            mesh: RodMesh,
            rigid_seg: RigidSegment,
            p0_target: np.ndarray,
            Q0_target: np.ndarray,
            f_ext_rigid: np.ndarray,
            tau_ext_rigid: np.ndarray,
            max_iter: int = 30,
            tol: float = 1e-6,
            lm_damping: float = 1e-3,
    ):
        self.mesh = mesh           # 包含 flex_seg 和 sigma_nodes/h_list
        self.flex = mesh.flex_seg
        self.rigid = rigid_seg
        self.p0_target = p0_target
        self.Q0_target = Q0_target
        self.f_ext_rigid = f_ext_rigid
        self.tau_ext_rigid = tau_ext_rigid
        self.max_iter = max_iter
        self.tol = tol
        self.lm_damping = lm_damping

        # 预先构造 C_S / C_BV 函数
        self.C_S_flex = make_C_S_flexible()
        self.C_BV_prox = make_C_BV_proximal_pose(p0_target, Q0_target)
        self.C_BV_dist = make_C_BV_distal_free_tip()

    # ---------- z 与块变量的互转 ----------
    # 现在的布局：
    # z = [ x_nodes( (M+1)*13 ), k_array( M*3*13 ), x_rigid(13) ]
    # 其中：
    #   x_nodes[n] = 柔性段 mesh 点 n 的状态 (n=0..M)
    #   k_array[n,i] = 区间 n, 第 i 个 stage 的斜率 (i=0..2)

    def pack_z(self, x_nodes: np.ndarray, k_array: np.ndarray, x_rigid: np.ndarray) -> np.ndarray:
        """
        x_nodes: shape (M+1, 13)
        k_array: shape (M, 3, 13)
        x_rigid: shape (13,)
        """
        return np.concatenate([x_nodes.reshape(-1), k_array.reshape(-1), x_rigid])

    def unpack_z(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回:
        x_nodes: (M+1, 13)
        k_array: (M, 3, 13)
        x_rigid: (13,)
        """
        M = self.mesh.M
        # 柔性 mesh 点
        x_nodes_flat_len = (M + 1) * 13
        x_nodes_flat = z[:x_nodes_flat_len]
        x_nodes = x_nodes_flat.reshape(M + 1, 13)

        # 斜率 k
        k_flat_len = M * 3 * 13
        k_flat = z[x_nodes_flat_len:x_nodes_flat_len + k_flat_len]
        k_array = k_flat.reshape(M, 3, 13)

        # 刚性段末端状态
        x_rigid = z[x_nodes_flat_len + k_flat_len:]
        return x_nodes, k_array, x_rigid

    # ---------- 全局残差 ----------

    def residual(self, z: np.ndarray) -> np.ndarray:
        x_nodes, k_array, x_rigid = self.unpack_z(z)
        M = self.mesh.M
        sigmas = self.mesh.sigma_nodes
        h_list = self.mesh.h_list

        E_list = []

        # --- 所有柔性段小区间 ---
        for n in range(M):
            x_n = x_nodes[n]
            x_np1 = x_nodes[n + 1]
            k_n = k_array[n]
            sigma_n = float(sigmas[n])
            h = float(h_list[n])

            # 每个区间是否带边界条件：
            if n == 0:
                # 第一个区间的入口是整体 proximal，施加基座 pose 约束
                def C_BV_flex(xn, xnp1, _fun=self.C_BV_prox):
                    return _fun(xn, xnp1)
            else:
                # 中间区间不施加边界条件
                def C_BV_flex(xn, xnp1):
                    return np.zeros(0)

            E_flex_n = self.flex.interval_residual_gl6(
                x_n=x_n,
                k_n=k_n,
                x_np1=x_np1,
                sigma_n=sigma_n,
                h=h,
                C_BV_fun=C_BV_flex,
                C_S_fun=self.C_S_flex,
            )
            E_list.append(E_flex_n)

        # --- 刚性段区间（整个刚性段视为一个“区间”） ---
        x_flex_end = x_nodes[-1]   # 柔性段末端 = 刚性段起点
        x2 = x_rigid               # 刚性段末端（未知）

        def C_BV_rigid(xn, xnp1, _fun=self.C_BV_dist):
            # 柔性→刚性连接处不加边界条件（只是内部节点），
            # 刚性段出口是整体 distal，施加零 wrench
            return _fun(xn, xnp1)

        E_rigid = self.rigid.interval_residual_rigid(
            x_n=x_flex_end,
            x_np1=x2,
            f_ext_total=self.f_ext_rigid,
            tau_ext_total=self.tau_ext_rigid,
            C_BV_fun=C_BV_rigid,
            C_S_fun=None,
        )
        E_list.append(E_rigid)

        return np.concatenate(E_list)

    # ---------- 数值 Jacobian (有限差分) ----------

    def jacobian_fd(self, z: np.ndarray, E: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        m = E.size
        n = z.size
        J = np.zeros((m, n))
        for j in range(n):
            z_pert = z.copy()
            z_pert[j] += eps
            E_pert = self.residual(z_pert)
            J[:, j] = (E_pert - E) / eps
        return J

    # ---------- LM 求解 ----------

    def solve(self, z0: np.ndarray) -> Tuple[np.ndarray, bool]:
        z = z0.copy()
        for it in range(self.max_iter):
            E = self.residual(z)
            normE = np.linalg.norm(E)
            print(f"[LM] iter={it}, ||E||={normE:.3e}")
            if normE < self.tol:
                return z, True

            J = self.jacobian_fd(z, E)
            A = J.T @ J + self.lm_damping * np.eye(J.shape[1])
            g = J.T @ E
            delta = np.linalg.solve(A, g)
            z = z - delta

        return z, False