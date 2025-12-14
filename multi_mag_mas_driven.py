from __future__ import annotations
from typing import List
import matplotlib.pyplot as plt
from pose_modules.equilibrium_solver import MultiSegmentEquilibriumSolver
from pose_modules.segments import RigidSegment, FlexibleSegment
from pose_modules.rod_mesh import RodMesh
from pose_modules.utils import *
from pose_modules.external_wrench import (
    GravityLineDensity,
    GravityRigid,
    make_external_wrench_density_flexible,
    compute_external_wrench_total_rigid,
)
from pose_modules.mas import (
    MagneticActuationSystem,
    SupieeMagneticModel,
)


def extract_catheter_points(
        meshes: List[RodMesh],
        rigid_segs: List[RigidSegment],
        x_nodes_list: List[np.ndarray],
        x_rigid_list: List[np.ndarray],
        gravity_rigid_list: List[GravityRigid],
        magnetic_model: SupieeMagneticModel,
        magnet_params_list: List[dict],
        coil_currents: np.ndarray,
        n_samples_rigid: int = 10,
) -> np.ndarray:
    """
    提取导管的3D点坐标，用于绘图

    返回：
    - P: (N_points, 3) 导管上所有采样点的坐标
    """
    N = len(rigid_segs)
    P_all = []

    for i in range(N):
        mesh_i = meshes[i]
        rigid_i = rigid_segs[i]
        x_nodes_i = x_nodes_list[i]
        gravity_rigid_i = gravity_rigid_list[i]
        magnet_params_i = magnet_params_list[i]

        # 柔性段所有网格点位置
        p_flex_i = x_nodes_i[:, :3]
        if i > 0:
            p_flex_i = p_flex_i[1:, :]
        P_all.append(p_flex_i)

        # 刚性段
        x_flex_end_i = x_nodes_i[-1]
        f_ext_total_i, tau_ext_total_i = compute_external_wrench_total_rigid(
            x_proximal=x_flex_end_i,
            rigid_length=rigid_i.length,
            gravity=gravity_rigid_i,
            magnetic_model=magnetic_model,
            magnet_params=magnet_params_i,
            coil_currents=coil_currents,
        )

        Lr = rigid_i.length
        sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)
        p_rigid_i = []
        for s in sigmas_rigid[1:]:
            x_s = rigid_i.state_along(
                x_proximal=x_flex_end_i,
                sigma=s,
                f_ext_total=f_ext_total_i,
                tau_ext_total=tau_ext_total_i,
            )
            p_rigid_i.append(x_s[:3])
        p_rigid_i = np.array(p_rigid_i)
        P_all.append(p_rigid_i)

    P = np.vstack(P_all)
    return P


def iteration_callback(iter_num, z, normE):
    """每次迭代的回调函数，每10步绘制一次导管姿态"""
    # 只在第0步和每10步时绘制
    if iter_num % 10 != 0:
        return

    # 解包状态
    x_nodes_list, k_array_list, x_rigid_list = solver.unpack_z(z)

    # 提取导管点
    P = extract_catheter_points(
        meshes=meshes,
        rigid_segs=rigid_segs,
        x_nodes_list=x_nodes_list,
        x_rigid_list=x_rigid_list,
        gravity_rigid_list=gravity_rigid_list,
        magnetic_model=supiee_model,
        magnet_params_list=magnet_params_list,
        coil_currents=coil_currents,
        n_samples_rigid=10,
    )

    # 存储历史
    iteration_history.append({
        'iter': iter_num,
        'P': P.copy(),
        'normE': normE
    })

    # 计算透明度：越接近收敛，透明度越低（颜色越深）
    # 使用对数尺度来映射透明度
    if normE > 1e-2:
        alpha = 0.9  # 初始阶段，高透明度
    elif normE > 1e-3:
        alpha = 0.6  # 中间阶段
    elif normE > 1e-4:
        alpha = 0.3  # 接近收敛
    else:
        alpha = 0.1  # 已收敛，低透明度（最深）

    # 颜色从浅蓝到深蓝
    color_intensity = 1.0 - alpha  # 透明度低时颜色深
    color = (0.2 * (1 - color_intensity), 0.4 * (1 - color_intensity), 0.8 + 0.2 * color_intensity)

    # 绘制当前迭代的导管
    ax.plot(P[:, 0], P[:, 1], P[:, 2],
            color=color, alpha=alpha, linewidth=1.5,
            label=f'iter={iter_num}, ||E||={normE:.2e}')

    # 更新标题
    ax.set_title(f'convergence history (iter={iter_num}, ||E||={normE:.3e})')

    # 刷新显示
    plt.draw()
    plt.pause(0.01)


def plot_catheter_3d_multiseg_with_magnetics(
        meshes: List[RodMesh],
        rigid_segs: List[RigidSegment],
        x_nodes_list_star: List[np.ndarray],
        x_rigid_list_star: List[np.ndarray],
        gravity_rigid_list: List[GravityRigid],
        magnetic_model: SupieeMagneticModel,
        magnet_params_list: List[dict],
        coil_currents: np.ndarray,
        n_samples_rigid: int = 10,
):
    """
    使用最终求解得到的 (x_nodes_list_star, x_rigid_list_star)，
    再次调用 compute_external_wrench_total_rigid 计算每个刚性段的总重力+磁力，
    然后用 RigidSegment.state_along 做采样，拼接整个 3F+3R 的空间曲线并绘制。
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    N = len(rigid_segs)
    P_all = []

    for i in range(N):
        mesh_i = meshes[i]
        rigid_i = rigid_segs[i]
        x_nodes_i = x_nodes_list_star[i]  # (Mi+1, 13)
        gravity_rigid_i = gravity_rigid_list[i]
        magnet_params_i = magnet_params_list[i]

        # 柔性段所有网格点位置
        p_flex_i = x_nodes_i[:, :3]
        # 避免重复连接点：除第一段外，去掉首个点
        if i > 0:
            p_flex_i = p_flex_i[1:, :]
        P_all.append(p_flex_i)

        # 当前刚性段的近端状态：柔性段末端
        x_flex_end_i = x_nodes_i[-1]

        # 在最终姿态下，计算该刚性段的“总外载”（重力 + 磁力）
        f_ext_total_i, tau_ext_total_i = compute_external_wrench_total_rigid(
            x_proximal=x_flex_end_i,
            rigid_length=rigid_i.length,
            gravity=gravity_rigid_i,
            magnetic_model=magnetic_model,
            magnet_params=magnet_params_i,
            coil_currents=coil_currents,
        )

        # 刚性段沿弧长采样
        Lr = rigid_i.length
        sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)

        p_rigid_i = []
        for s in sigmas_rigid[1:]:  # 跳过 s=0
            x_s = rigid_i.state_along(
                x_proximal=x_flex_end_i,
                sigma=s,
                f_ext_total=f_ext_total_i,
                tau_ext_total=tau_ext_total_i,
            )
            p_rigid_i.append(x_s[:3])
        p_rigid_i = np.array(p_rigid_i)
        P_all.append(p_rigid_i)

    P = np.vstack(P_all)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:, 0], P[:, 1], P[:, 2], marker='o', markersize=3, linewidth=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multi-segment catheter (gravity + magnetic) configuration')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])

    # 视野范围，你可以按实际长度调整
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0, 0.15])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 材料 & 几何参数
    d_outer = 1.5e-3  # m

    # 柔性段长度
    Lf1 = 0.05
    Lf2 = 0.03
    Lf3 = 0.01

    # NdFeB 刚体段长度
    Lr1 = Lr2 = Lr3 = 0.03

    # 柔性段刚度矩阵
    K_se_pdms, K_bt_pdms = build_k_matrices_for_pdms(d_outer)
    print(f"K_se_pdms:\n{K_se_pdms}; K_bt_pdms:\n{K_bt_pdms}")

    # 构造柔性段
    flex1 = FlexibleSegment(length=Lf1, K_se=K_se_pdms, K_bt=K_bt_pdms)
    flex2 = FlexibleSegment(length=Lf2, K_se=K_se_pdms, K_bt=K_bt_pdms)
    flex3 = FlexibleSegment(length=Lf3, K_se=K_se_pdms, K_bt=K_bt_pdms)

    # 柔性段重力线密度
    grav_line = build_gravity_line_density_for_pdms(d_outer)
    fext1, tauext1 = make_external_wrench_density_flexible(gravity=grav_line)
    fext2, tauext2 = make_external_wrench_density_flexible(gravity=grav_line)
    fext3, tauext3 = make_external_wrench_density_flexible(gravity=grav_line)

    flex1.fext_density = fext1
    flex1.tauext_density = tauext1
    flex2.fext_density = fext2
    flex2.tauext_density = tauext2
    flex3.fext_density = fext3
    flex3.tauext_density = tauext3

    # 构建柔性段的 Gauss-Legendre 区间网格
    mesh1 = RodMesh(flex_seg=flex1, n_intervals=3)
    mesh2 = RodMesh(flex_seg=flex1, n_intervals=3)
    mesh3 = RodMesh(flex_seg=flex1, n_intervals=3)

    # 构造刚性段
    rigid1 = RigidSegment(length=Lr1, v_star=np.array([0.0, 0.0, 1.0]))
    rigid2 = RigidSegment(length=Lr2, v_star=np.array([0.0, 0.0, 1.0]))
    rigid3 = RigidSegment(length=Lr3, v_star=np.array([0.0, 0.0, 1.0]))

    flex_segs = [flex1, flex2, flex3]
    meshes = [mesh1, mesh2, mesh3]
    rigid_segs = [rigid1, rigid2, rigid3]

    # 刚性段重力参数
    gravity_rigid_list = [
        build_gravity_rigid_for_ndfeb(d_outer, Lr1),
        build_gravity_rigid_for_ndfeb(d_outer, Lr2),
        build_gravity_rigid_for_ndfeb(d_outer, Lr3),
    ]

    # Supiee: 磁力模型 & 磁场参数 & 输入电流
    calib_file = "calib/mpem_calibration_file_sp=40_order=1.yaml"
    mas = MagneticActuationSystem(calib_file=calib_file)
    supiee_model = SupieeMagneticModel(mas=mas)

    # 电流: 幅值 50A 以内的随机电流
    coil_currents = np.random.rand(8) * 50.0

    # 磁性参数
    m_mag = 0.005301  # A·m^2 磁偶极矩
    magnet_params_list = [
        {"m_body": np.array([0.0, 0.0, m_mag])},   # R1
        {"m_body": np.array([0.0, 0.0, m_mag])},   # R2
        {"m_body": np.array([0.0, 0.0, m_mag])},  # R3
    ]

    # 额外手动外载: 暂不考虑, 置零
    N_pairs = 3
    f_ext_list = [np.zeros(3) for _ in range(N_pairs)]
    tau_ext_list = [np.zeros(3) for _ in range(N_pairs)]


    # 边界条件
    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    # 初始猜测
    z0, x_node_list0, k_array_list0, x_rigid_list0 = make_initial_guess_multi(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
    )

    # 求解器构造
    solver = MultiSegmentEquilibriumSolver(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_list=f_ext_list,
        tau_ext_list=tau_ext_list,
        gravity_rigid_list=gravity_rigid_list,
        magnetic_model=supiee_model,
        magnet_params_list=magnet_params_list,
        coil_currents=coil_currents,
        max_iter=20000,
        tol=1e-6,
        lm_damping=1e-4,
    )

    # 迭代过程中可视化回调
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('导管收敛过程可视化 (每10步更新)')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0, 0.15])

    plt.ion()  # 开启交互模式
    plt.show(block=False)

    iteration_history = []  # 存储迭代历史

    z_star, success = solver.solve(z0, callback=iteration_callback)
    print("Multi-segment LM success:", success)

    x_nodes_list_star, k_array_list_star, x_rigid_list_star = solver.unpack_z(z_star)

    for i in range(N_pairs):
        p_flex_end = x_nodes_list_star[i][-1, :3]
        p_rigid_end = x_rigid_list_star[i][:3]
        print(f"Segment pair {i+1}:")
        print("  F end p =", p_flex_end)
        print("  R end p =", p_rigid_end)

    plt.ioff()

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 5:
        # 只显示前3个和最后2个
        selected_handles = handles[:3] + handles[-2:]
        selected_labels = labels[:3] + labels[-2:]
        ax.legend(selected_handles, selected_labels, loc='upper right', fontsize=8)
    else:
        ax.legend(loc='upper right', fontsize=8)

    print(f"\n迭代过程可视化完成！共绘制了 {len(iteration_history)} 个迭代状态")
    plt.show()

    # 可选：绘制单独的最终姿态图
    if success:
        plot_catheter_3d_multiseg_with_magnetics(
            meshes=meshes,
            rigid_segs=rigid_segs,
            x_nodes_list_star=x_nodes_list_star,
            x_rigid_list_star=x_rigid_list_star,
            gravity_rigid_list=gravity_rigid_list,
            magnetic_model=supiee_model,
            magnet_params_list=magnet_params_list,
            coil_currents=coil_currents,
            n_samples_rigid=20,
        )







