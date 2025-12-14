from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Protocol
import numpy as np
from .segments import quat_to_rotmat


# -------- 1. 数据结构定义 --------


@dataclass
class GravityLineDensity:
    """
    定义:
    - rhoA: float 柔性段重力线密度(N/m)
    - g_vec: np.ndarray 世界坐标系下的重力方向, 通常定义为 [0, 0, -1]
    """
    rhoA: float
    g_vec: np.ndarray


    def force_world(self) -> np.ndarray:
        v = np.asarray(self.g_vec, dtype=float).reshape(3)
        return self.rhoA * v


@dataclass
class GravityRigid:
    """
    定义刚体段的重力参数:
    - mass: float 刚体段总质量 (kg)
    - g_world: np.ndarray 世界坐标系下的重力加速度 (m/s^2)
    - r_cm_body: np.ndarray 质心在刚体局部坐标中的位置 (3,)
    """
    mass: float
    g_world: np.ndarray
    r_cm_body: np.ndarray


    def total_force_world(self) -> np.ndarray:
        # G = m * g
        return self.mass * self.g_world.reshape(3)

    def total_torque_body(self) -> np.ndarray:
        """
        如果需要在刚体局部算合力矩, 可以用 r_cm × (R^T F_world),
        :return: np.ndarray
        """
        raise NotImplementedError("Use world-frame torque computation with segment pose.")


# --------- 2. 场域解析模型接口 ---------


class MagneticModel(Protocol):
    """
    抽象接口, 用于计算Supiee系统作用于磁体的磁力和磁力矩

    定义:
    - position_world: 磁体几何中心在世界坐标中的位置
    - R_world_from_body: 3x3 旋转矩阵，刚体局部 -> 世界
    - magnet_params: 被动磁体参数
    - coil_currents: 线圈电流向量
    """

    def wrench_on_magnet(
            self,
            position_world: np.ndarray,
            R_world_from_body: np.ndarray,
            magnet_params: dict,
            coil_currents: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return F_world: (3,) 世界坐标磁力
        :return Tau_world: (3,) 世界坐标磁力矩
        """
        ...


# -------- 3. 柔性段外力/外力矩密度 --------


def make_external_wrench_density_flexible(
        gravity: Optional[GravityLineDensity] = None,
        magnetic_density_fun: Optional[
            Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]
        ] = None,
):
    """
    生成 (fext_density, tauext_density) 两个函数, 给 FlexibleSegment 使用

    参数
    ----
    gravity : GravityLineDensity or None
        若不为 None, 则在整段上施加世界坐标系下的均匀重力线密度, 并转到杆坐标。
    magnetic_density_fun : callable or None
        若不为 None, 则应接受 (x, sigma) -> (f_mag_body, tau_mag_body)
        其中:
          - x: 当前状态向量 (13,)
          - sigma: 当前弧长位置
        返回:
          - f_mag_body : (3,) 在杆局部坐标中的力密度
          - tau_mag_body : (3,) 在杆局部坐标中的力矩密度
    """

    def fext_density(x: np.ndarray, sigma: float) -> np.ndarray:
        """
        杆坐标系下的柔性段外力密度
        """

        # 获取当前姿态 R
        Q = x[3:7]
        Q = Q / np.linalg.norm(Q)
        R = quat_to_rotmat(Q)

        f_total_body = np.zeros(3)

        # 1) 重力: world -> body
        if gravity is not None:
            f_g_world = gravity.force_world()  # (3,)
            f_g_body = R.T @ f_g_world
            f_total_body += f_g_body

        # 2) 磁力密度: 柔性段一般不受到磁力作用
        # 但如果给出, 直接认为 magnetic_density_fun 已经给的是 body frame
        if magnetic_density_fun is not None:
            f_mag_body, _ = magnetic_density_fun(x, sigma)
            f_total_body += f_mag_body

        return f_total_body


    def tauext_density(x: np.ndarray, sigma: float) -> np.ndarray:
        """
        杆坐标系下的柔性段外力矩密度
        """

        tau_total_body = np.zeros(3)

        # 对于均匀质量杆, 重力分布作用在杆中心线上, 不会产生外力矩

        # 磁力矩密度: 柔性段一般不受到磁力作用
        # 但如果给出, 直接认为 magnetic_density_fun 已经给的是 body frame
        if magnetic_density_fun is not None:
            _, tau_mag_body = magnetic_density_fun(x, sigma)
            tau_total_body += tau_mag_body

        return tau_total_body

    return fext_density, tauext_density


# -------- 4. 刚性段受到集中外力/外力矩 --------


def compute_external_wrench_total_rigid(
        x_proximal: np.ndarray,
        rigid_length: float,
        gravity: Optional[GravityRigid] = None,
        magnetic_model: Optional[MagneticModel] = None,
        magnet_params: Optional[dict] = None,
        coil_currents: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算世界坐标系下的刚性段的总外力和总外力矩

    参数
    ----
    x_proximal : (13,)
        该刚性段近端状态, 与前一段末端一致
        其中:
          p(0) = x_proximal[:3]       # 近端位置
          Q(0) = x_proximal[3:7]     # 近端姿态
    rigid_length : float
        该刚性段长度 Lr
    gravity : GravityRigid or None
        若不为 None, 则为刚体段加上重力总合力/合力矩
    magnetic_model : MagneticModel or None
        若不为 None, 则调用 magnetic_model.wrench_on_magnet(...)
        计算该刚体磁体的磁力/磁力矩
    magnet_params : dict or None
        传给 magnetic_model 的磁体参数 (体积/磁化强度等)
    coil_currents : np.ndarray or None
        线圈电流向量, 传给 magnetic_model

    返回
    ----
    f_ext_total_world : (3,)
        总合力(世界坐标)
    tau_ext_total_world : (3,)
        总合力矩(世界坐标, 关于刚性段近端点)
    """

    p0 = x_proximal[:3]
    Q0 = x_proximal[3:7]
    Q0 = Q0 / np.linalg.norm(Q0)
    R0 = quat_to_rotmat(Q0)  # world_from_body

    f_total_world = np.zeros(3)
    tau_total_world = np.zeros(3)

    # 1) 重力: 作用在质心
    # 合力 Fg = m*g
    # 力矩 tau_g = r_cm_world x Fg
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

    # 2) 磁力/磁力矩, 假定磁体几何中心在刚性段中点
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

        # 若 Tau_world_about_mid 是关于中点的力矩,
        # 则关于近端点 p0 的总力矩 = (p_mid_world - p0) × Fm_world + Tau_world_about_mid
        r_mid = p_mid_world - p0
        tau_m_world = np.cross(r_mid, Fm_world) + Tau_world_about_mid

        f_total_world += Fm_world
        tau_total_world += tau_m_world

    return f_total_world, tau_total_world