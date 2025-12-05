# external_wrench.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Protocol

import numpy as np

from segments import quat_to_rotmat, skew


# ========= 1. 一些参数结构 =========

@dataclass
class GravityLineDensity:
    """
    柔性段用的重力参数：线密度（N/m） + 世界坐标重力方向
    例如：rhoA = 0.01 N/m, g_vec = [0, 0, -1] 表示沿 -Z 方向 0.01N/m
    通常你可以设 rhoA = mass_per_length * 9.81, g_vec = [0, 0, -1]
    """
    rhoA: float  # N/m, 已经乘了 9.81
    g_vec: np.ndarray  # (3,), 单位方向或直接 N/m 向量

    def force_world(self) -> np.ndarray:
        v = np.asarray(self.g_vec, dtype=float).reshape(3)
        # 如果 g_vec 是单位向量，可以 rhoA * g_vec；否则你也可以直接把 g_vec 当“每米力”
        return self.rhoA * v


@dataclass
class GravityRigid:
    """
    刚体段用的重力参数：
    - mass: 刚体段总质量 (kg)
    - g_world: 世界坐标系重力加速度 (m/s^2)
    - r_cm_body: 质心在刚体局部坐标中的位置 (3,)
    """
    mass: float
    g_world: np.ndarray
    r_cm_body: np.ndarray

    def total_force_world(self) -> np.ndarray:
        # F = m * g
        return self.mass * self.g_world.reshape(3)

    def total_torque_body(self) -> np.ndarray:
        """
        如果需要在刚体局部算合力矩，可以用 r_cm × (R^T F_world)，
        不过通常我们在 world 系下处理即可。
        这里留一个接口占位，用时再细化。
        """
        raise NotImplementedError("Use world-frame torque computation with segment pose.")


# ========= 2. 磁力模型接口（预留给 Supiee / FE 场） =========

class MagneticModel(Protocol):
    """
    一个抽象接口，用来从 Supiee / FE 场得到磁力和力矩。
    你后续可以用 magnetic_actuation_system.MagneticActuationSystem
    来实现这个接口。

    假定：
    - position_world: 磁体几何中心在世界坐标中的位置
    - R_world_from_body: 3x3 旋转矩阵，刚体局部 -> 世界
    - magnet_params: 任意你需要的参数（磁体体积、磁化强度、方向等）
    - coil_currents: 线圈电流向量（例如 shape (8,)）
    """

    def wrench_on_magnet(
            self,
            position_world: np.ndarray,
            R_world_from_body: np.ndarray,
            magnet_params: dict,
            coil_currents: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回:
        - F_world: (3,) 世界坐标磁力
        - Tau_world: (3,) 世界坐标磁力矩（关于 position_world）
        """
        ...


# ========= 3. 柔性段：external_wrench_density =========

def make_external_wrench_density_flexible(
        gravity: Optional[GravityLineDensity] = None,
        magnetic_density_fun: Optional[
            Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]
        ] = None,
):
    """
    生成 (fext_density, tauext_density) 两个函数，给 FlexibleSegment 使用。

    参数
    ----
    gravity : GravityLineDensity or None
        若不为 None，则在整段上施加均匀重力线密度（世界坐标），并转到杆坐标。
    magnetic_density_fun : callable or None
        若不为 None，则应接受 (x, sigma) -> (f_mag_body, tau_mag_body)
        其中:
          - x: 当前状态向量 (13,)
          - sigma: 当前弧长位置
        返回:
          - f_mag_body : (3,) 在杆局部坐标中的力密度
          - tau_mag_body : (3,) 在杆局部坐标中的力矩密度

        将来你如果想把 FE + Supiee 的结果插值到连续弧长上，
        可以在外部封装一个类似的 fun。
    """

    def fext_density(x: np.ndarray, sigma: float) -> np.ndarray:
        """
        柔性段外力密度（杆坐标系下）。
        """
        # 拿到当前姿态 R
        Q = x[3:7]
        Q = Q / np.linalg.norm(Q)
        R = quat_to_rotmat(Q)  # world_from_body

        f_total_body = np.zeros(3)

        # 1) 重力：world -> body
        if gravity is not None:
            f_g_world = gravity.force_world()  # (3,)
            f_g_body = R.T @ f_g_world
            f_total_body += f_g_body

        # 2) 磁力密度：直接认为 magnetic_density_fun 已经给的是 body frame
        if magnetic_density_fun is not None:
            f_mag_body, _ = magnetic_density_fun(x, sigma)
            f_total_body += f_mag_body

        return f_total_body

    def tauext_density(x: np.ndarray, sigma: float) -> np.ndarray:
        """
        柔性段外力矩密度(杆坐标系下)
        """
        tau_total_body = np.zeros(3)

        # 重力通常作用在中心线附近，对细长杆可以近似忽略重力引起的外力矩密度，
        # 若你有更精确的模型，可以在这里加上。
        # 例如 tau_g_body = r_offset_body × f_g_body

        # 磁力矩密度
        if magnetic_density_fun is not None:
            _, tau_mag_body = magnetic_density_fun(x, sigma)
            tau_total_body += tau_mag_body

        return tau_total_body

    return fext_density, tauext_density


# ========= 4. 刚性段：external_wrench_total =========

def compute_external_wrench_total_rigid(
        x_proximal: np.ndarray,
        rigid_length: float,
        gravity: Optional[GravityRigid] = None,
        magnetic_model: Optional[MagneticModel] = None,
        magnet_params: Optional[dict] = None,
        coil_currents: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单个刚性段上的总合力 f_ext_total 和总合力矩 tau_ext_total（世界坐标）。

    参数
    ----
    x_proximal : (13,)
        该刚性段近端状态（与前一段末端一致）。
        其中:
          p(0) = x_proximal[:3]       # 近端位置
          Q(0) = x_proximal[3:7]     # 近端姿态（四元数）
    rigid_length : float
        该刚性段长度 Lr
    gravity : GravityRigid or None
        若不为 None，则为刚体段加上重力总合力/合力矩。
    magnetic_model : MagneticModel or None
        若不为 None，则调用 magnetic_model.wrench_on_magnet(...)
        计算该刚体磁体的磁力/磁力矩。
    magnet_params : dict or None
        传给 magnetic_model 的磁体参数（体积、磁化强度等）。
    coil_currents : np.ndarray or None
        线圈电流向量，传给 magnetic_model。

    返回
    ----
    f_ext_total_world : (3,)
        总合力（世界坐标）
    tau_ext_total_world : (3,)
        总合力矩（世界坐标, 关于刚性段近端点）
    """
    p0 = x_proximal[:3]
    Q0 = x_proximal[3:7]
    Q0 = Q0 / np.linalg.norm(Q0)
    R0 = quat_to_rotmat(Q0)  # world_from_body

    f_total_world = np.zeros(3)
    tau_total_world = np.zeros(3)

    # 1) 重力：作用在质心，合力 Fg = m g，力矩 tau_g = r_cm_world × Fg
    if gravity is not None:
        Fg_world = gravity.total_force_world()  # (3,)
        # 质心在 body 中位置 -> world
        r_cm_body = gravity.r_cm_body.reshape(3)
        r_cm_world = R0 @ r_cm_body + p0  # position of CM in world
        # 力矩关于近端点 p0: tau = (r_cm_world - p0) × Fg_world
        r_vec = r_cm_world - p0
        tau_g_world = np.cross(r_vec, Fg_world)

        f_total_world += Fg_world
        tau_total_world += tau_g_world

    # 2) 磁力 / 磁力矩：假定磁体几何中心在刚性段中点
    if magnetic_model is not None and magnet_params is not None and coil_currents is not None:
        # 刚性段中点在 body 中: [0, 0, Lr/2]
        r_mid_body = np.array([0.0, 0.0, rigid_length / 2.0])
        p_mid_world = R0 @ r_mid_body + p0  # world 中磁体中心

        Fm_world, Tau_world_about_mid = magnetic_model.wrench_on_magnet(
            position_world=p_mid_world,
            R_world_from_body=R0,
            magnet_params=magnet_params,
            coil_currents=coil_currents,
        )

        # 若 Tau_world_about_mid 是关于中点的力矩，
        # 则关于近端点 p0 的总力矩 = (p_mid_world - p0) × Fm_world + Tau_world_about_mid
        r_mid = p_mid_world - p0
        tau_m_world = np.cross(r_mid, Fm_world) + Tau_world_about_mid

        f_total_world += Fm_world
        tau_total_world += tau_m_world

    return f_total_world, tau_total_world
