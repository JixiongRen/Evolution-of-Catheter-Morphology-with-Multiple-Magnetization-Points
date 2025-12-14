from mag_manip.mag_manip import ForwardModelMPEM
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from dataclasses import dataclass
from .external_wrench import MagneticModel


class MagneticActuationSystem():
    def __init__(self, calib_file: str):
        self.calib_file = calib_file
        self.mas_forward_model = ForwardModelMPEM()
        self.mas_forward_model.setCalibrationFile(self.calib_file)

    def b_field(self, position: np.ndarray, currents_vector: np.ndarray):
        """
        计算给定位置由一组电流产生的磁场

        参数
        ----------
        position : numpy.array of shape (3,)
            磁场将在其计算的三维空间位置
        currents_vector : numpy.array of shape (n,),
            将用于计算磁场的电流向量

        返回
        -------
        numpy.array of shape (3,)
            在给定位置计算的磁场
        """
        return self.mas_forward_model.computeFieldFromCurrents(position, currents_vector).flatten()

    def b_field_gradient(self, position: np.ndarray, currents_vector: np.ndarray):
        """
        计算给定位置由一组电流产生的磁场梯度

        参数
        ----------
        position : numpy.array of shape (3,)
            磁场将在其计算的三维空间位置
        currents_vector : numpy.array of shape (n,),
            将用于计算磁场的电流向量

        返回
        -------
        numpy.array of shape (5,)
            在给定位置计算的磁场梯度
        """
        return self.mas_forward_model.computeGradient5FromCurrents(position, currents_vector).flatten()

    def magnetic_wrench(self, pose_list: list, magnetic_moment: np.ndarray, currents_vector: np.ndarray):
        """
        给定一组磁体姿态, 磁体参数, 电流向量; 计算目标磁体在背景磁场下受到的磁力和磁力矩

        参数
        ----------
        pose_list : list of numpy.array of shape (3, n)
            磁性参数的位姿列表
        magnetic_moment : numpy.array of shape (3, n)
            受力磁体的磁矩向量
        currents_vector : numpy.array of shape (n,)
            将用于产生背景磁场的电流向量

        返回
        -------
        numpy.array of shape (6, n)
            n个磁性位点在背景磁场下受到的电磁力和电磁力矩
        """
        # 将输入统一为 (3, n) 形式
        if isinstance(pose_list, np.ndarray):
            if pose_list.ndim == 1:
                positions = pose_list.reshape(3, 1)
            else:
                positions = pose_list
        else:
            # pose_list 是由形状为 (3,) 的 numpy.array 组成的列表
            positions = np.stack(pose_list, axis=1)

        if magnetic_moment.ndim == 1:
            magnetic_moment = magnetic_moment.reshape(3, 1)

        n = positions.shape[1]
        wrench = np.zeros((6, n), dtype=np.float64)

        for i in range(n):
            pos = positions[:, i]
            m = magnetic_moment[:, i]

            # 磁场和磁场梯度
            B = self.b_field(pos, currents_vector)  # [Bx, By, Bz]
            G5 = self.b_field_gradient(pos, currents_vector)  # [dBxdx, dBxdy, dBxdz, dBydy, dBzdz]

            mx, my, mz = m

            # 根据公式 F_m = (m · ∇)B
            force_matrix = np.array([
                [mx, my, mz, 0, 0],
                [0, mx, 0, my, mz],
                [-mz, 0, mx, -mz, my],
            ], dtype=np.float64)

            Fm = force_matrix @ G5

            # 根据公式 T_m = m × B
            torque_matrix = np.array([
                [0, -mz, my],
                [mz, 0, -mx],
                [-my, mx, 0],
            ], dtype=np.float64)

            Tm = torque_matrix @ B

            wrench[:3, i] = Fm
            wrench[3:, i] = Tm

        return wrench

    def visualize_magnetic_wrench(self, pose_list: list, magnetic_moment: np.ndarray, currents_vector: np.ndarray):
        """
        在包含所有磁性位点的一条平面上，可视化磁场模值等高线，
        并将三个磁性位点绘制为小矩形，同时叠加力和力矩在该平面的投影箭头。

        参数
        ----------
        pose_list : numpy.array of shape (3, n)
            磁性参数的位姿列表（这里假定 n=3 且不共线）
        magnetic_moment : numpy.array of shape (3, n)
            受力磁体的磁矩向量
        currents_vector : numpy.array of shape (n_coils,)
            将用于产生背景磁场的电流向量
        """
        # 统一为 (3, n) 形式
        if isinstance(pose_list, np.ndarray):
            positions = pose_list
        else:
            positions = np.stack(pose_list, axis=1)

        # 取前三个点定义平面
        p0 = positions[:, 0]
        p1 = positions[:, 1]
        p2 = positions[:, 2]

        v1 = p1 - p0
        v2 = p2 - p0

        # 构造平面上的正交基 e1, e2
        e1 = v1 / (np.linalg.norm(v1) + 1e-12)
        v2_proj = v2 - np.dot(v2, e1) * e1
        e2 = v2_proj / (np.linalg.norm(v2_proj) + 1e-12)

        # 将三个位点投影到 (u, v) 坐标系
        rel = positions.T - p0  # (n, 3)
        u_coords = rel @ e1
        v_coords = rel @ e2

        # 在 (u, v) 平面上定义网格范围，留一定边界
        margin_u = 0.02
        margin_v = 0.02
        umin, umax = u_coords.min() - margin_u, u_coords.max() + margin_u
        vmin, vmax = v_coords.min() - margin_v, v_coords.max() + margin_v

        Nu = Nv = 80
        us = np.linspace(umin, umax, Nu)
        vs = np.linspace(vmin, vmax, Nv)
        U, V = np.meshgrid(us, vs, indexing="ij")  # 形状 (Nu, Nv)

        # 平面上所有网格点对应的三维坐标
        points = p0[None, :] + U.reshape(-1, 1) * e1[None, :] + V.reshape(-1, 1) * e2[None, :]

        # 计算每个网格点的磁场并取模值（列表推导避免多重 for）
        B_list = [self.b_field(p, currents_vector) for p in points]
        B = np.vstack(B_list)  # (Nu*Nv, 3)
        B_mag = np.linalg.norm(B, axis=1).reshape(U.shape)

        # 计算三个位点的受力/力矩
        wrench = self.magnetic_wrench(positions, magnetic_moment, currents_vector)
        forces = wrench[:3, :]
        torques = wrench[3:, :]

        # 将力和力矩投影到 (u, v) 平面
        F_u = forces.T @ e1
        F_v = forces.T @ e2
        T_u = torques.T @ e1
        T_v = torques.T @ e2

        fig, ax = plt.subplots(figsize=(8, 6))

        # 磁场模值等高线
        contour = ax.contourf(U, V, B_mag, levels=30, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="|B| (T)")

        # 绘制磁性位点为小矩形
        rect_w = (umax - umin) * 0.03
        rect_h = (vmax - vmin) * 0.03
        for i in range(positions.shape[1]):
            ui, vi = u_coords[i], v_coords[i]
            rect = plt.Rectangle((ui - rect_w / 2.0, vi - rect_h / 2.0),
                                 rect_w, rect_h,
                                 edgecolor="black", facecolor="red", alpha=0.8)
            ax.add_patch(rect)

        # 在平面上绘制力和力矩箭头（投影后的 2D 向量）
        scale_F = 0.3 * max(umax - umin, vmax - vmin)
        scale_T = 0.3 * max(umax - umin, vmax - vmin)

        for i in range(positions.shape[1]):
            ui, vi = u_coords[i], v_coords[i]

            fu, fv = F_u[i], F_v[i]
            tu, tv = T_u[i], T_v[i]

            # 归一化后再按尺度放大
            F_len = np.hypot(fu, fv) or 1.0
            T_len = np.hypot(tu, tv) or 1.0

            ax.arrow(ui, vi,
                     (fu / F_len) * scale_F * 0.1,
                     (fv / F_len) * scale_F * 0.1,
                     color="white", width=0.0005,
                     length_includes_head=True,
                     label="Force" if i == 0 else None)

            ax.arrow(ui, vi,
                     (tu / T_len) * scale_T * 0.1,
                     (tv / T_len) * scale_T * 0.1,
                     color="yellow", width=0.0005,
                     length_includes_head=True,
                     label="Torque" if i == 0 else None)

        ax.set_xlabel("u (plane axis 1)")
        ax.set_ylabel("v (plane axis 2)")
        ax.set_title("|B| contour and projected wrench on plane through 3 sites")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()


@dataclass
class SupieeMagneticModel(MagneticModel):
    """
    用 Supiee 的 MagneticActuationSystem 实现 external_wrench.MagneticModel 接口

    约定 magnet_params 至少包含:
    - "m_body": np.ndarray shape (3,), 刚体坐标系下的磁矩向量
    """

    mas: MagneticActuationSystem

    def wrench_on_magnet(
            self,
            position_world: np.ndarray,
            R_world_from_body: np.ndarray,
            magnet_params: Dict,
            coil_currents: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1) 取出刚体坐标系下的磁矩 m_body
        m_body = np.asarray(magnet_params["m_body"]).reshape(3)

        # 2) 转到世界坐标系
        m_world = R_world_from_body @ m_body

        # 3) 调用你原来的磁力/力矩接口
        #    magnetic_wrench 接受 (3, n) 形式, 这里 n = 1
        pose = position_world.reshape(3, 1)
        m_vec = m_world.reshape(3, 1)
        wrench = self.mas.magnetic_wrench(pose_list=pose,
                                          magnetic_moment=m_vec,
                                          currents_vector=coil_currents)
        # wrench shape (6,1)
        F_world = wrench[:3, 0]
        Tau_world = wrench[3:, 0]
        return F_world, Tau_world