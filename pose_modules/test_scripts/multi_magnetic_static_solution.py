# multi_segment_equilibrium_solver.py

from __future__ import annotations
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from pose_modules.equilibrium_solver import MultiSegmentEquilibriumSolver
from pose_modules.segments import RigidSegment
from pose_modules.rod_mesh import RodMesh
from pose_modules.utils import (
    make_initial_guess_multi,
)


def plot_catheter_3d_multiseg(
    meshes: List[RodMesh],
    rigid_segs: List[RigidSegment],
    x_nodes_list_star: List[np.ndarray],
    x_rigid_list_star: List[np.ndarray],
    f_ext_list: List[np.ndarray],
    tau_ext_list: List[np.ndarray],
    n_samples_rigid: int = 10,
):
    """Plot the final multi-segment catheter centerline."""
    N = len(rigid_segs)
    P_all = []

    for i in range(N):
        mesh_i = meshes[i]
        rigid_i = rigid_segs[i]
        x_nodes_i = x_nodes_list_star[i]  # (Mi+1, 13)

        p_flex_i = x_nodes_i[:, :3]
        if i > 0:
            p_flex_i = p_flex_i[1:, :]
        P_all.append(p_flex_i)

        x_flex_end_i = x_nodes_i[-1]
        Lr = rigid_i.length
        sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)

        p_rigid_i = []
        for s in sigmas_rigid[1:]:
            x_s = rigid_i.state_along(
                x_proximal=x_flex_end_i,
                sigma=s,
                f_ext_total=f_ext_list[i],
                tau_ext_total=tau_ext_list[i],
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
    ax.set_title('Multi-segment catheter final configuration')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([0, 0.25])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pose_modules.segments import FlexibleSegment, RigidSegment
    from pose_modules.rod_mesh import RodMesh

    flex1 = FlexibleSegment(length=0.06, K_se=np.eye(3), K_bt=np.eye(3))
    flex2 = FlexibleSegment(length=0.06, K_se=np.eye(3), K_bt=np.eye(3))
    flex3 = FlexibleSegment(length=0.06, K_se=np.eye(3), K_bt=np.eye(3))

    mesh1 = RodMesh(flex_seg=flex1, n_intervals=4)
    mesh2 = RodMesh(flex_seg=flex2, n_intervals=4)
    mesh3 = RodMesh(flex_seg=flex3, n_intervals=4)

    rigid1 = RigidSegment(length=0.01, v_star=np.array([0., 0., 1.]))
    rigid2 = RigidSegment(length=0.01, v_star=np.array([0., 0., 1.]))
    rigid3 = RigidSegment(length=0.01, v_star=np.array([0., 0., 1.]))

    flex_segs = [flex1, flex2, flex3]
    meshes = [mesh1, mesh2, mesh3]
    rigid_segs = [rigid1, rigid2, rigid3]

    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    f_ext_list = [
        np.array([3.4, -3.7, 0.4]) * 1e-3,
        np.array([-8.5, 2.1, 3.3]) * 1e-3,
        np.array([6.0, -1.3, -2.1]) * 1e-3,
    ]
    tau_ext_list = [
        np.array([-18.9, 4.7, 13.4]),
        np.array([-1.3, -19.1, 8.9]),
        np.array([19.4, 0.7, 12.6]),
    ]

    z0, x_nodes_list0, k_array_list0, x_rigid_list0 = make_initial_guess_multi(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
    )

    solver = MultiSegmentEquilibriumSolver(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_list=f_ext_list,
        tau_ext_list=tau_ext_list,
        max_iter=200000,
        tol=1e-2,
        lm_damping=1e-3,
    )

    z_star, success = solver.solve(z0)
    print("Multi-segment LM success:", success)

    x_nodes_list_star, k_array_list_star, x_rigid_list_star = solver.unpack_z(z_star)

    for i in range(3):
        p_flex_end = x_nodes_list_star[i][-1, :3]
        p_rigid_end = x_rigid_list_star[i][:3]
        print(f"Segment pair {i+1}:")
        print("  F end p =", p_flex_end)
        print("  R end p =", p_rigid_end)

    if success:
        plot_catheter_3d_multiseg(
            meshes=meshes,
            rigid_segs=rigid_segs,
            x_nodes_list_star=x_nodes_list_star,
            x_rigid_list_star=x_rigid_list_star,
            f_ext_list=f_ext_list,
            tau_ext_list=tau_ext_list,
            n_samples_rigid=20,
        )
