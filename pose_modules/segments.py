from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from .basics import (
    skew,
    quat_to_rotmat,
    quat_derivative,
    GL3_A,
    GL3_B,
    GL3_C,
)

# -------- 1. Segment 基类 --------


@dataclass
class Segment:
    """
    segment - 包括柔性段/刚体(磁体)段
    公共基类, 仅存储几何长度等信息
    """
    length: float
    name: str = ""

    def propagate(self, *args, **kwargs):
        """ 子类必须实现在给定近端状态和外载下, 沿着段弧长sigma (0 <= sigma <= L) 传播状态的函数 """
        raise NotImplementedError


# -------- 2. 柔性段 FlexibleSegment --------


@dataclass
class FlexibleSegment(Segment):
    """
    flexible segment - 柔性段
    """
    K_se: np.ndarray = np.eye(3)
    K_bt: np.ndarray = np.eye(3)
    v_star: np.ndarray = np.array([0., 0., 1.])
    u_star: np.ndarray = np.array([0., 0., 0.])

    # 外载荷注入: 重力/磁场力/外部接触力
    fext_density: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    tauext_density: Optional[Callable[[np.ndarray, float], np.ndarray]] = None


    def cosserat_rhs(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """
        实现cosserat杆的右端项g(x(sigma), sigma)

        参数
        ----------
        x : numpy.array of shape (13,)
            13x1 状态向量 [p(3), Q(4), f(3), tau(3)]^T
        sigma : float
            当前沿段的局部弧长位置 sigma ∈ [0, L]

        返回
        -----
        numpy.array of shape (13,)
            13*1 状态向量 [p(3), Q(4), f(3), tau(3)]^T 对 sigma 的导数
        """

        # 拆状态
        p = x[:3]
        Q = x[3:7]
        f = x[7:10]
        tau = x[10:13]

        # 保证 Q 归一化
        Q = Q / np.linalg.norm(Q)

        R = quat_to_rotmat(Q)

        # 外力/外力矩密度
        if self.fext_density is None:
            f_bar = np.zeros(3)
        else:
            f_bar = self.fext_density(x, sigma)

        if self.tauext_density is None:
            tau_bar = np.zeros(3)
        else:
            tau_bar = self.tauext_density(x, sigma)

        # 线应变 / 角应变
        Kse_inv = np.linalg.inv(self.K_se)
        Kbt_inv = np.linalg.inv(self.K_bt)

        v = Kse_inv @ (R.T @ f) + self.v_star
        u = Kbt_inv @ (R.T @ tau) + self.u_star

        # 各状态量的微分
        # dp/dsigma = R(Q) * v
        dp_dsigma = R @ v

        # dQ/dsigma = 1/2 * [ -q^T; qr I - [q]_x ] * omega
        omega_global = R @ u
        dQ_dsigma = quat_derivative(Q, omega_global)

        # df/dsigma = f_bar
        df_dsigma = - f_bar

        # dtau/dsigma = [f]_x dp/dsigma - \bar τ_ext(x, σ)
        df_x = skew(f)
        dtau_dsigma = df_x @ dp_dsigma - tau_bar

        # 组装 dx/dsigma
        dx_sigma = np.zeros_like(x)
        dx_sigma[:3] = dp_dsigma
        dx_sigma[3:7] = dQ_dsigma
        dx_sigma[7:10] = df_dsigma
        dx_sigma[10:13] = dtau_dsigma

        return dx_sigma


    def propagate(self,
                  x_proximal: np.ndarray,
                  integrator: Callable[[Callable, np.ndarray, float, float], np.ndarray],
                  n_steps: int = 1) -> np.ndarray:
        """
        从近端状态x(0)=x_proximal积分到sigma=L, 返回远端状态x(L)

        integrator: 单步积分器, 入参为 (rhs, x0, s0, s1)


        参数
        ----------
        x : numpy.array of shape (13,)
            13*1 状态向量 [p(3), Q(4), f(3), tau(3)]^T
        sigma : float
            当前沿段的局部弧长位置 sigma ∈ [0, L]

        返回
        -----
        numpy.array of shape (13,)
            13*1 状态向量 [p(3), Q(4), f(3), tau(3)]^T 对 sigma 的导数
        """

        x = x_proximal.copy()
        s0 = 0.
        s1 = self.length
        if n_steps <= 0:
            n_steps = 1
        h = (s1 - s0) / n_steps

        sigma = s0
        for _ in range(n_steps):
            x = integrator(self.cosserat_rhs, x, sigma, sigma + h)
            sigma += h

        return x


    def interval_residual_gl6(
            self,
            x_n: np.ndarray,  # 区间起点状态 x_n
            k_n: np.ndarray,  # shape(3, 13). 3个stage斜率 k_{n, 1..3}
            x_np1: np.ndarray,  # 区间终点状态 x_{n+1}
            sigma_n: float,  # 当前小区间起点弧长 sigma_n
            h: float,  # 当前小区间长度 h
            C_BV_fun=None,  # 可选: 边值约束函数
            C_S_fun=None,  # 可选: 状态约束函数: 单位四元数/接触函数
    ) -> np.ndarray:
        """
        计算 GL6 求解器的残差

        参数
        ----------
        x_n : numpy.array of shape (13,)
            当前小区间起点状态 x_n
        k_n : numpy.array of shape (3, 13)
            当前小区间 3 个 stage 的斜率 k_{n, 1..3}
        x_np1 : numpy.array of shape (13,)
            当前小区间终点状态 x_{n+1}
        sigma_n : float
            当前小区间起点弧长 sigma_n
        h : float
            当前小区间长度 h
        C_BV_fun : callable, optional
            边值约束函数
        C_S_fun : callable, optional
            状态约束函数: 单位四元数/接触函数

        返回
        -----
        numpy.array
            当前小区间残差 E(z_n)
            E(z_n) = [ C_S ;
                   x_{n+1} - x_n - h Σ b_i k_i ;
                   k_i - g(x_stage_i, sigma_stage_i) for i=1..3 ;
                   C_BV ]
        """

        # 1. C_S: 状态约束
        if C_S_fun is None:
            C_S = np.zeros(0)
        else:
            C_S = C_S_fun(x_n, x_np1)

        # 2. 状态连续性约束
        res_state = x_np1 - x_n - h * sum(
            GL3_B[i] * k_n[i] for i in range(3)
        )

        # 3. 三个stage的ODE一致性: k_i-g(x_stage_i, σ_n + c_i h)
        res_ks = []
        for i in range(3):
            # x_stage_i = x_n + h Σ_j a_ij k_j
            x_stage_i = x_n.copy()
            for j in range(3):
                x_stage_i = x_stage_i + h * GL3_A[i, j] * k_n[j]

            sigma_stage_i = sigma_n + GL3_C[i] * h

            # g(x, σ) = cosserat_rhs(x, σ)
            g_i = self.cosserat_rhs(x_stage_i, sigma_stage_i)

            res_ki = k_n[i] - g_i
            res_ks.append(res_ki)

        res_ks = np.concatenate(res_ks, axis=0)  # 3*13 维

        # 4. C_BV: 如果该区间是接触边界, 即 distal 或 proximal, 则需要添加边界约束
        if C_BV_fun is None:
            C_BV = np.zeros(0)
        else:
            C_BV = C_BV_fun(x_n, x_np1)

        # 5. 段残差
        E_n = np.concatenate([C_S, res_state, res_ks, C_BV])
        return E_n


# -------- 3. 刚性段 --------


@dataclass
class RigidSegment(Segment):
    """
    刚性段(永磁体段)类实现

    这里假设
    - 段为直线, 无内禀曲率
    - v* 给出杆轴在杆坐标系中的方向, 默认为 [0,0,1]^T
    - 外载在本段上均匀分布, 其总合力/合力矩分别为 f_ext, tau_ext
    """

    v_star: np.ndarray = np.array([0., 0., 1.])
    f_ext: np.ndarray = np.zeros(3)
    tau_ext: np.ndarray = np.zeros(3)

    def state_along(
            self,
            x_proximal: np.ndarray,
            sigma: float,
            f_ext_total: np.ndarray,
            tau_ext_total: np.ndarray
    ) -> np.ndarray:
        """
        计算给定近端状态和外载下, 沿着段 sigma ∈ [0, L] 传播状态的函数

        参数
        ----------
        x_proximal : numpy.array of shape (13,)
            近端状态
        sigma : float
            当前弧长位置 sigma ∈ [0, L]
        f_ext_total : numpy.array of shape (3,)
            总合力
        tau_ext_total : numpy.array of shape (3,)
            总合力矩

        返回
        -----
        numpy.array of shape (13,)
            当前弧长位置 sigma ∈ [0, L] 的状态 x(sigma)
        """

        L = self.length

        # 限制 sigma 在 [0, L] 范围内
        sigma = float(np.clip(sigma, 0., L))

        pp = x_proximal[:3]
        Qp = x_proximal[3:7]
        fp = x_proximal[7:10]
        tau_p = x_proximal[10:13]

        Qp = Qp / np.linalg.norm(Qp)
        R_p = quat_to_rotmat(Qp)

        # ----- 位置和姿态 -----
        p_sigma = pp + sigma * (R_p @ self.v_star)
        Q_sigma = Qp.copy()  # 刚体段中, 姿态不随 sigma 变化

        # ----- 内力 f(sigma) -----
        f_sigma = fp - (sigma / L) * f_ext_total

        # ----- 内力矩 tau(sigma) -----
        r = p_sigma - pp
        term1 = tau_p - (sigma / L) * tau_ext_total
        term2 = - skew(r) @ fp
        term3 = (sigma / (2. * L)) * (skew(r) @ f_ext_total)
        tau_sigma = term1 + term2 + term3

        # 组回状态向量
        x_sigma = np.zeros_like(x_proximal)
        x_sigma[:3] = p_sigma
        x_sigma[3:7] = Q_sigma
        x_sigma[7:10] = f_sigma
        x_sigma[10:13] = tau_sigma

        return x_sigma

    def propagate(
            self,
            x_proximal: np.ndarray,
            f_ext_total: np.ndarray,
            tau_ext_total: np.ndarray,
    ) -> np.ndarray:
        """
        直接给出远端 sigma = L 处的状态 x(L)，是 state_along 的特殊情况
        """
        return self.state_along(
            x_proximal=x_proximal,
            sigma=self.length,
            f_ext_total=f_ext_total,
            tau_ext_total=tau_ext_total,
        )

    def interval_residual_rigid(
            self,
            x_n: np.ndarray,  # 刚性段起点状态
            x_np1: np.ndarray,  # 刚性段终点状态
            f_ext_total: np.ndarray,  # 总合力
            tau_ext_total: np.ndarray,  # 总合力矩
            C_BV_fun=None,  # 可选: 边值约束函数
            C_S_fun=None,  # 可选: 若还有别的状态约束, 例如接触
    ) -> np.ndarray:
        """
        刚性段上单个区间的误差 E_n
        - 用刚体解析 T(x_n) 直接算出 x_rigid(L)
        - C_S 包含 x_{n+1} - x_rigid(L)、单位四元数等
        - 不存在有限差分的状态/斜率项
        """
        # 1. 解析刚体变换求预测状态 x_rigid(L)
        x_rigid_end = self.propagate(x_n, f_ext_total, tau_ext_total)

        # 2. C_S: 刚体关系 + 单位四元数约束
        C_S_core = x_np1 - x_rigid_end
        Q_np1 = x_np1[3:7]
        quat_norm_res = np.array([np.dot(Q_np1, Q_np1) - 1.])  # 四元数单位约束

        C_S_list = [C_S_core, quat_norm_res]

        if C_S_fun is not None:
            C_S_list.append(C_S_fun(x_n, x_np1))

        C_S = np.concatenate(C_S_list)

        # 3. C_BV: 若该刚性段起点/末端恰好是整根导管的边界
        if C_BV_fun is None:
            C_BV = np.zeros(0)
        else:
            C_BV = C_BV_fun(x_n, x_np1)

        E_n = np.concatenate([C_S, C_BV])

        return E_n