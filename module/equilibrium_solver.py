import numpy as np
from typing import Tuple, List
from segments import RigidSegment, FlexibleSegment
from catheter import RodMesh
from utils import make_C_S_flexible, make_C_BV_proximal_pose, make_C_BV_distal_free_tip
from dataclasses import dataclass

class SingleSegmentEquilibriumSolver:
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


@dataclass
class MultiSegmentEquilibriumSolver:
    """
    多段 Cosserat 导管静稳态求解器：支持 N 段柔性 + N 段刚性串联
    结构：base -> F1 -> R1 -> F2 -> R2 -> ... -> FN -> RN

    刚性段外载通过 f_ext_list[i], tau_ext_list[i] 施加在 Ri 上，
    不做磁场耦合，只是给定每段刚体的总合力/合力矩。
    """

    flex_segs: List[FlexibleSegment]
    meshes: List[RodMesh]              # 与 flex_segs 一一对应
    rigid_segs: List[RigidSegment]     # 与 flex_segs 一一对应
    p0_target: np.ndarray              # (3,) 基座位置
    Q0_target: np.ndarray              # (4,) 基座姿态（四元数）
    f_ext_list: List[np.ndarray]       # 每段刚体上的总合力 (3,)
    tau_ext_list: List[np.ndarray]     # 每段刚体上的总合力矩 (3,)
    max_iter: int = 30
    tol: float = 1e-6
    lm_damping: float = 1e-3

    def __post_init__(self):
        assert len(self.flex_segs) == len(self.meshes) == len(self.rigid_segs)
        assert len(self.rigid_segs) == len(self.f_ext_list) == len(self.tau_ext_list)
        self.N = len(self.flex_segs)

        # 每个柔性段的区间数 M_i
        self.M_list = [mesh.M for mesh in self.meshes]

        # 预先构造“公共”的状态约束/边界约束
        self.C_S_flex = make_C_S_flexible()
        self.C_BV_prox = make_C_BV_proximal_pose(self.p0_target, self.Q0_target)
        self.C_BV_dist = make_C_BV_distal_free_tip()

        # 预计算整体 z 的长度，方便 debug
        self._z_dim_nodes = sum((M + 1) * 13 for M in self.M_list)
        self._z_dim_ks = sum(M * 3 * 13 for M in self.M_list)
        self._z_dim_rigids = self.N * 13
        self._z_dim_total = self._z_dim_nodes + self._z_dim_ks + self._z_dim_rigids

    # ---------- z 与块变量的互转 ----------
    # 约定：
    # z = [ x_nodes_F1, ..., x_nodes_FN, k_F1, ..., k_FN, x_R1, ..., x_RN ]
    # 其中：
    #   x_nodes_Fi : shape (Mi+1, 13)
    #   k_Fi       : shape (Mi, 3, 13)
    #   x_Ri       : shape (13,)

    def pack_z(
        self,
        x_nodes_list: List[np.ndarray],   # 每个 (Mi+1, 13)
        k_array_list: List[np.ndarray],   # 每个 (Mi, 3, 13)
        x_rigid_list: List[np.ndarray],   # 每个 (13,)
    ) -> np.ndarray:
        pieces = []
        # 柔性节点
        for x_nodes in x_nodes_list:
            pieces.append(x_nodes.reshape(-1))
        # 柔性区间斜率
        for k_array in k_array_list:
            pieces.append(k_array.reshape(-1))
        # 刚性末端
        for xR in x_rigid_list:
            pieces.append(xR.reshape(-1))

        z = np.concatenate(pieces)
        return z

    def unpack_z(
        self,
        z: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        返回：
        - x_nodes_list: [ (M1+1,13), ..., (MN+1,13) ]
        - k_array_list: [ (M1,3,13), ..., (MN,3,13) ]
        - x_rigid_list: [ (13,), ..., (13,) ]
        """
        cursor = 0
        x_nodes_list: List[np.ndarray] = []
        k_array_list: List[np.ndarray] = []
        x_rigid_list: List[np.ndarray] = []

        # 1) 拆 x_nodes
        for i, M in enumerate(self.M_list):
            n_flat = (M + 1) * 13
            block = z[cursor:cursor + n_flat]
            cursor += n_flat
            x_nodes = block.reshape(M + 1, 13)
            x_nodes_list.append(x_nodes)

        # 2) 拆 k_array
        for i, M in enumerate(self.M_list):
            k_flat_len = M * 3 * 13
            block = z[cursor:cursor + k_flat_len]
            cursor += k_flat_len
            k_array = block.reshape(M, 3, 13)
            k_array_list.append(k_array)

        # 3) 拆 x_rigid
        for _ in range(self.N):
            xR = z[cursor:cursor + 13]
            cursor += 13
            x_rigid_list.append(xR)

        assert cursor == z.size
        return x_nodes_list, k_array_list, x_rigid_list

    # ---------- 全局残差 ----------

    def residual(self, z: np.ndarray) -> np.ndarray:
        """
        拼接所有柔性区间 + 刚性段的残差，得到全局 E(z)
        """
        x_nodes_list, k_array_list, x_rigid_list = self.unpack_z(z)
        E_list = []

        # --------- 柔性段残差 ---------
        for i in range(self.N):
            flex = self.flex_segs[i]
            mesh = self.meshes[i]
            x_nodes = x_nodes_list[i]
            k_array = k_array_list[i]
            M = mesh.M
            sigmas = mesh.sigma_nodes
            h_list = mesh.h_list

            for n in range(M):
                x_n = x_nodes[n]
                x_np1 = x_nodes[n + 1]
                k_n = k_array[n]
                sigma_n = float(sigmas[n])
                h = float(h_list[n])

                # C_BV_flex
                #  - F1 第一个区间入口: 基座 pose 约束
                #  - 其它柔性段 Fi (i>0) 的第一个区间入口: 与前一刚性段 Ri-1 的末端状态一致
                #  - 其它区间不加边界条件
                if i == 0 and n == 0:
                    # 近端固定
                    def C_BV_flex(xn, xnp1, _fun=self.C_BV_prox):
                        return _fun(xn, xnp1)
                elif i > 0 and n == 0:
                    # 接到前一刚体 Ri-1 的末端
                    xR_prev = x_rigid_list[i - 1]

                    def C_BV_flex(xn, xnp1, x_target=xR_prev):
                        # 这里用 13 维状态整体连续性。
                        return xn - x_target
                else:
                    # 中间区间/柔性段内部，不施加边界条件
                    def C_BV_flex(xn, xnp1):
                        return np.zeros(0)

                E_flex_n = flex.interval_residual_gl6(
                    x_n=x_n,
                    k_n=k_n,
                    x_np1=x_np1,
                    sigma_n=sigma_n,
                    h=h,
                    C_BV_fun=C_BV_flex,
                    C_S_fun=self.C_S_flex,
                )
                E_list.append(E_flex_n)

        # --------- 刚性段残差 ---------
        for i in range(self.N):
            rigid = self.rigid_segs[i]
            f_ext = self.f_ext_list[i]
            tau_ext = self.tau_ext_list[i]

            # Ri 的近端是 Fi 的末端
            x_flex_end_i = x_nodes_list[i][-1]
            xR_i = x_rigid_list[i]

            # 只有最后一个刚性段 RN 才施加 distal 自由端边界条件
            if i == self.N - 1:
                def C_BV_rigid(xn, xnp1, _fun=self.C_BV_dist):
                    return _fun(xn, xnp1)
            else:
                # 中间刚性段的出口是内部节点，后续由下一个柔性段入口的 C_BV 接上
                def C_BV_rigid(xn, xnp1):
                    return np.zeros(0)

            E_rigid_i = rigid.interval_residual_rigid(
                x_n=x_flex_end_i,
                x_np1=xR_i,
                f_ext_total=f_ext,
                tau_ext_total=tau_ext,
                C_BV_fun=C_BV_rigid,
                C_S_fun=None,
            )
            E_list.append(E_rigid_i)

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
            print(f"[Multi-LM] iter={it}, ||E||={normE:.3e}")
            if normE < self.tol:
                return z, True

            J = self.jacobian_fd(z, E)
            A = J.T @ J + self.lm_damping * np.eye(J.shape[1])
            g = J.T @ E
            delta = np.linalg.solve(A, g)
            z = z - delta

        return z, False