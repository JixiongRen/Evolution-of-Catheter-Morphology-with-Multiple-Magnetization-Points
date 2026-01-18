
from __future__ import annotations

import os, sys, json, argparse
from typing import List, Optional, Tuple, Dict, Any

# Allow running as a standalone script from this directory (no package install needed).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nondim import NondimScales, x_bar_to_dim
from basics_nondim import quat_normalize
from segments_nondim import FlexibleParams, RigidParams
from external_wrench_nondim import GravityRigid, MagneticModel, compute_external_wrench_total_rigid
from equilibrium_solver_nondim import (
    SolverParams,
    MultiSegmentEquilibriumSolverNondimJAX,
    print_top_blocks_jax,   # 新增：直接打印Top blocks
)
from utils_nondim import (
    make_initial_guess_multi_bar_jax,
    unpack_z_bar_jax,
    build_k_matrices_circular_jax,
    build_gravity_line_density_jax,
    build_gravity_rigid_jax,
)
from rod_mesh_nondim import UniformMesh, build_uniform_mesh
from mas_nondim import MagneticActuationSystem

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


# ------------------- Parsing helpers -------------------
def _parse_csv_floats(s: str) -> List[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).replace(";", ",").split(",") if str(x).strip() != ""]


def _parse_csv_ints(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(float(x)) for x in str(s).replace(";", ",").split(",") if str(x).strip() != ""]


def _parse_vec3(s: str) -> Array:
    vals = _parse_csv_floats(s)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got {len(vals)} from: {s}")
    return jnp.array(vals, dtype=jnp.float64)


def _parse_quat(s: str) -> Array:
    vals = _parse_csv_floats(s)
    if len(vals) != 4:
        raise ValueError(f"Expected 4 floats for quaternion [w,x,y,z], got {len(vals)} from: {s}")
    return quat_normalize(jnp.array(vals, dtype=jnp.float64))


def _parse_list_vec3(s: str) -> List[Array]:
    # Format: "x,y,z; x,y,z; x,y,z"
    if s is None or str(s).strip() == "":
        return []
    out = []
    for item in str(s).split(";"):
        item = item.strip()
        if not item:
            continue
        out.append(_parse_vec3(item))
    return out


def _repeat_or_validate(vals: List[Any], n: int, *, name: str) -> List[Any]:
    if len(vals) == 0:
        raise ValueError(f"{name} is empty.")
    if len(vals) == 1 and n > 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"{name} length mismatch: expected {n}, got {len(vals)}.")
    return vals



# ------------------- Advancer / protruding length -------------------
from advancer_nondim import apply_advancer_protrude_length

# ------------------- Magnetic Model adapter -------------------
class SupieeRigidMagneticModelJAX(MagneticModel):
    """Adapter to satisfy external_wrench_nondim_jax.MagneticModel protocol."""

    def __init__(self, mas: MagneticActuationSystem):
        self.mas = mas

    def compute_wrench_world(
        self,
        p_world: Array,
        R_world_from_body: Array,
        magnet_params: Dict,
        coil_currents: Array,
    ) -> Tuple[Array, Array]:
        p_world = jnp.asarray(p_world, dtype=jnp.float64).reshape(3,)
        R_world_from_body = jnp.asarray(R_world_from_body, dtype=jnp.float64).reshape(3, 3)
        coil_currents = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(8,)

        m_body = jnp.asarray(magnet_params["m_body"], dtype=jnp.float64).reshape(3,)
        m_world = R_world_from_body @ m_body

        pose = p_world.reshape(3, 1)
        m_vec = m_world.reshape(3, 1)

        wrench_6x1 = self.mas.magnetic_wrench(pose_list=pose, magnetic_moment=m_vec, currents_vector=coil_currents)
        f_world = wrench_6x1[0:3, 0]
        tau_world = wrench_6x1[3:6, 0]
        return f_world, tau_world


# ------------------- Scales -------------------
def compute_scales_from_flex(
    *,
    L_ref: float,
    d_outer: float,
    E: float,
    G: float,
) -> NondimScales:
    """Pragmatic scales:
      - L_ref: user-specified (typically total catheter length)
      - F_ref: max diag(K_se) ~= max(GA, EA)
      - M_ref: F_ref * L_ref
    """
    K_se, _ = build_k_matrices_circular_jax(d_outer=d_outer, E=E, G=G)
    F_ref = float(jnp.max(jnp.diag(K_se)))
    M_ref = float(F_ref * float(L_ref))
    return NondimScales(L_ref=float(L_ref), F_ref=float(F_ref), M_ref=float(M_ref))


# ------------------- Build solver params -------------------
def build_solver_params(
    *,
    # segment counts: N == len(flex_lengths) == len(rigid_lengths) == len(M_list)
    flex_lengths: List[float],
    rigid_lengths: List[float],
    M_list: List[int],
    # flexible material / geometry per segment (allow singleton or per-seg)
    flex_d_outer: List[float],
    flex_E: List[float],
    flex_G: List[float],
    flex_rho: List[float],
    # rigid material / geometry per segment
    rigid_d_outer: List[float],
    rigid_rho: List[float],
    # kinematics
    scales: NondimScales,
    p0_dim: Array,
    Q0: Array,
    axis_body: Array,
    # environment
    enable_gravity: bool,
    g_world: Array,
    # magnetics
    enable_magnetics: bool,
    calib_file: Optional[str],
    actuation_table_pkl: Optional[str],
    coil_currents: Optional[Array],
    m_body_list: Optional[List[Array]],
) -> Tuple[SolverParams, List[UniformMesh]]:
    assert len(flex_lengths) == len(rigid_lengths) == len(M_list)
    N = len(flex_lengths)

    flex_d_outer = _repeat_or_validate(flex_d_outer, N, name="flex_d_outer")
    flex_E = _repeat_or_validate(flex_E, N, name="flex_E")
    flex_G = _repeat_or_validate(flex_G, N, name="flex_G")
    flex_rho = _repeat_or_validate(flex_rho, N, name="flex_rho")

    rigid_d_outer = _repeat_or_validate(rigid_d_outer, N, name="rigid_d_outer")
    rigid_rho = _repeat_or_validate(rigid_rho, N, name="rigid_rho")

    # Meshes (uniform per segment)
    meshes: List[UniformMesh] = [build_uniform_mesh(float(flex_lengths[i]), int(M_list[i])) for i in range(N)]

    # Axis / pose
    Q0 = quat_normalize(jnp.asarray(Q0, dtype=jnp.float64).reshape(4,))
    axis_body = jnp.asarray(axis_body, dtype=jnp.float64).reshape(3,)
    axis_body = axis_body / jnp.maximum(jnp.linalg.norm(axis_body), 1e-12)

    # Flexible params (per segment)
    flex_params: List[FlexibleParams] = []
    for i in range(N):
        K_se, K_bt = build_k_matrices_circular_jax(d_outer=float(flex_d_outer[i]), E=float(flex_E[i]), G=float(flex_G[i]))
        flex_params.append(
            FlexibleParams(
                length=float(flex_lengths[i]),
                Kse_inv=jnp.linalg.inv(jnp.asarray(K_se, dtype=jnp.float64)),
                Kbt_inv=jnp.linalg.inv(jnp.asarray(K_bt, dtype=jnp.float64)),
                v_star=axis_body,              # body-frame axial direction
                u_star=jnp.zeros((3,), dtype=jnp.float64),
            )
        )

    # Rigid params (per segment)
    rigid_params: List[RigidParams] = []
    for i in range(N):
        rigid_params.append(
            RigidParams(
                length=float(rigid_lengths[i]),
                v_star=axis_body,              # body-frame axial direction
            )
        )

    # Proximal BC target (bar)
    p0_bar = jnp.asarray(p0_dim, dtype=jnp.float64).reshape(3,) / float(scales.L_ref)

    # Flexible distributed loads: gravity line density in WORLD frame (SI)
    if enable_gravity:
        g_world = jnp.asarray(g_world, dtype=jnp.float64).reshape(3,)
        flex_f_line_world_list = tuple(
            build_gravity_line_density_jax(rho=float(flex_rho[i]), d_outer=float(flex_d_outer[i]), g_world=g_world).force_world()
            for i in range(N)
        )
    else:
        flex_f_line_world_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))
    flex_tau_line_world_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))

    # Rigid lumped loads: gravity + magnetics computed in residual
    gravity_rigid_list: List[Optional[GravityRigid]] = []
    for i in range(N):
        if enable_gravity:
            gravity_rigid_list.append(
                build_gravity_rigid_jax(
                    rho=float(rigid_rho[i]),
                    d_outer=float(rigid_d_outer[i]),
                    length=float(rigid_lengths[i]),
                    g_world=jnp.asarray(g_world, dtype=jnp.float64),
                    v_star_body=axis_body,
                )
            )
        else:
            gravity_rigid_list.append(None)

    # Additional user-provided totals (kept as zeros unless you need them)
    f_ext_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))
    tau_ext_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))

    # Magnetics setup
    magnetic_model = None
    magnet_params_list: List[Optional[Dict]] = [None for _ in range(N)]
    coil_currents_out = None

    if enable_magnetics:
        if calib_file is None:
            raise ValueError("enable_magnetics=True but calib_file is None.")
        if coil_currents is None:
            raise ValueError("enable_magnetics=True but coil_currents is None.")
        if m_body_list is None:
            raise ValueError("enable_magnetics=True but m_body_list is None.")
        if len(m_body_list) != N:
            raise ValueError(f"m_body_list length mismatch: expected {N}, got {len(m_body_list)}")

        mas = MagneticActuationSystem(
            calib_file=calib_file,
            actuation_table_pkl=actuation_table_pkl,
            dtype=jnp.float32,
            enable_x64=True,
        )
        magnetic_model = SupieeRigidMagneticModelJAX(mas)
        coil_currents_out = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(8,)

        for i in range(N):
            magnet_params_list[i] = {"m_body": jnp.asarray(m_body_list[i], dtype=jnp.float64).reshape(3,)}

    # Residual layout (MUST match equilibrium_solver_nondim_jax.residual_bar stacking)
    cs_len = 2
    flex_block_offsets: List[int] = []
    flex_block_lens: List[int] = []
    rigid_block_offsets: List[int] = []
    rigid_block_lens: List[int] = []
    offset = 0

    for i in range(N):
        M = int(M_list[i])
        for n in range(M):
            if (i == 0) and (n == 0):
                bv_len = 7    # proximal pose constraint
            elif (i > 0) and (n == 0):
                bv_len = 13   # connect to previous rigid
            else:
                bv_len = 0
            ln = cs_len + 13 + 39 + bv_len
            flex_block_offsets.append(offset)
            flex_block_lens.append(ln)
            offset += ln

    for i in range(N):
        bv_len = 6 if (i == N - 1) else 0  # free tip at final rigid
        ln = 13 + 1 + bv_len
        rigid_block_offsets.append(offset)
        rigid_block_lens.append(ln)
        offset += ln

    params = SolverParams(
        flex=tuple(flex_params),
        rigid=tuple(rigid_params),
        sbar_nodes=tuple(m.sbar_nodes for m in meshes),
        hbar_list=tuple(m.hbar_list for m in meshes),
        M_list=tuple(int(m.M) for m in meshes),
        p0_bar=p0_bar,
        Q0=Q0,
        f_ext_list=f_ext_list,
        tau_ext_list=tau_ext_list,
        flex_f_line_world_list=tuple(flex_f_line_world_list),
        flex_tau_line_world_list=tuple(flex_tau_line_world_list),
        gravity_rigid_list=tuple(gravity_rigid_list),
        magnet_params_list=tuple(magnet_params_list),
        coil_currents=coil_currents_out,
        magnetic_model=magnetic_model,
        scales=scales,
        cs_len=int(cs_len),
        flex_block_offsets=tuple(int(x) for x in flex_block_offsets),
        flex_block_lens=tuple(int(x) for x in flex_block_lens),
        rigid_block_offsets=tuple(int(x) for x in rigid_block_offsets),
        rigid_block_lens=tuple(int(x) for x in rigid_block_lens),
        total_E_len=int(offset),
    )
    return params, meshes


# ------------------- Visualization -------------------
def plot_converged_pose_3d(
    z_bar: Array,
    params: SolverParams,
    scales: NondimScales,
    *,
    n_samples_rigid: int = 12,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    x_nodes_list_bar, _, _ = unpack_z_bar_jax(z_bar, M_list=params.M_list)
    N = len(params.flex)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    any_plotted = False

    from segments_nondim import rigid_state_along_dim

    for i in range(N):
        x_nodes_bar = jnp.asarray(x_nodes_list_bar[i])  # (M+1, 13)
        p_flex_m = np.asarray(x_nodes_bar[:, 0:3] * float(scales.L_ref))

        if p_flex_m.size:
            ax.plot(p_flex_m[:, 0], p_flex_m[:, 1], p_flex_m[:, 2], "-o", markersize=2.5, linewidth=1.5)
            any_plotted = True

        rigidp = params.rigid[i]
        Lr = float(rigidp.length)
        if Lr <= 0.0 or n_samples_rigid <= 0:
            continue

        x_prox_dim = x_bar_to_dim(x_nodes_bar[-1], scales)

        f_total_dim, tau_total_dim = compute_external_wrench_total_rigid(
            x_proximal=x_prox_dim,
            rigid_length=Lr,
            gravity=params.gravity_rigid_list[i],
            magnetic_model=params.magnetic_model,
            magnet_params=params.magnet_params_list[i],
            coil_currents=params.coil_currents,
        )

        sigmas = np.linspace(0.0, Lr, int(n_samples_rigid) + 1)[1:]
        p_rigid = []
        for s in sigmas:
            x_s = rigid_state_along_dim(
                x_prox_dim=x_prox_dim,
                sigma=float(s),
                rigid=rigidp,
                f_ext_total_dim=f_total_dim,
                tau_ext_total_dim=tau_total_dim,
            )
            p_rigid.append(np.asarray(x_s[0:3]))
        if p_rigid:
            pr = np.asarray(p_rigid)
            ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], "-o", markersize=2.5, linewidth=1.5)
            any_plotted = True

    if not any_plotted:
        print("[plot] No points to plot.")
        plt.close(fig)
        return

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Converged catheter pose (meters)")
    ax.set_xlim([-0.10, 0.10])
    ax.set_ylim([-0.10, 0.10])
    ax.set_zlim([-0.10, 0.10])
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-segment catheter equilibrium (JAX nondim) with CLI-configurable parameters.")
    p.add_argument("--config", type=str, default=None, help="Optional JSON config file. CLI flags override config.")
    p.add_argument("--flex_lengths", type=str, default="0.03,0.03,0.03", help="Comma-separated flexible lengths [m]. If --L_protrude is set, you may provide either N values (L1 placeholder + L2..LN) or N-1 values (L2..LN).")
    p.add_argument("--rigid_lengths", type=str, default="0.003,0.003,0.003", help="Comma-separated rigid lengths [m].")
    # Advancer (sheath protruding length)
    p.add_argument(
        "--L_protrude",
        type=float,
        default=None,
        help=(
            "Total length protruding from the sheath exit [m]. If set, the first flexible segment length is inferred "
            "so that sum(flex_lengths)+sum(rigid_lengths)=L_protrude, while all other segment lengths remain fixed. "
            "You may provide flex_lengths as either N values (L1 placeholder + L2..LN) or N-1 values (L2..LN)."
        ),
    )
    p.add_argument(
        "--L1_min",
        type=float,
        default=1e-6,
        help=(
            "Minimum allowed inferred first flexible segment length [m] when using --L_protrude. "
            "If inferred L1_new < L1_min, an exception is raised."
        ),
    )

    p.add_argument("--M_list", type=str, default="5,5,5", help="Comma-separated intervals per flexible segment.")

    # Geometry/material (allow singleton -> broadcast)
    p.add_argument("--flex_d_outer", type=str, default="0.0015", help="Flexible outer diameters [m], comma-separated or single value.")
    p.add_argument("--flex_E", type=str, default="1.8e6", help="Flexible Young's modulus [Pa], comma-separated or single value.")
    p.add_argument("--flex_G", type=str, default="0.6e6", help="Flexible shear modulus [Pa], comma-separated or single value.")
    p.add_argument("--flex_rho", type=str, default="970.0", help="Flexible density [kg/m^3], comma-separated or single value.")

    p.add_argument("--rigid_d_outer", type=str, default="0.0015", help="Rigid outer diameters [m], comma-separated or single value.")
    p.add_argument("--rigid_rho", type=str, default="7500.0", help="Rigid density [kg/m^3], comma-separated or single value.")

    # Pose / axis
    p.add_argument("--p0", type=str, default="0,0,-0.05", help="Proximal position p0 in WORLD frame [m], format x,y,z.")
    p.add_argument("--Q0", type=str, default="1,0,0,0", help="Proximal quaternion [w,x,y,z] (WORLD_from_BODY).")
    p.add_argument("--axis_body", type=str, default="0,0,1", help="Centerline axis direction in BODY frame (unit), format x,y,z.")

    # Gravity
    p.add_argument("--enable_gravity", action="store_true", help="Enable gravity (default OFF unless set).")
    p.add_argument("--g_world", type=str, default="0,0,-9.81", help="Gravity vector in WORLD frame [m/s^2], format x,y,z.")

    # Magnetics
    p.add_argument("--enable_magnetics", action="store_true", help="Enable magnetics (default OFF unless set).")
    p.add_argument("--calib_file", type=str, default=None, help="Calibration file path for Supiee MAS.")
    p.add_argument("--actuation_table_pkl", type=str, default=None, help="Optional actuation_table.pkl path.")
    p.add_argument("--coil_currents", type=str, default=None, help="8-coil currents, comma-separated.")
    p.add_argument("--m_body_list", type=str, default=None, help="Per-segment magnet moments in BODY frame; format 'x,y,z; x,y,z; ...'")

    # Solver
    p.add_argument("--max_iter", type=int, default=100000)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--lm_damping", type=float, default=1e-1)
    p.add_argument("--jac_method", type=str, default="fwd", choices=["fwd", "rev"])
    p.add_argument("--plot", action="store_true", help="Plot converged pose (matplotlib).")
    return p


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_iteration_callback(
    params: SolverParams,
    scales: NondimScales,
    *,
    every: int = 10,
    n_samples_rigid: int = 10,
    save_every: int = 100,
    save_dir: Optional[str] = None,
    save_npz: bool = True,
):
    """
    在 LM 迭代过程中周期性（every 步）绘制当前姿态：
      - 柔性段：不透明蓝色（节点，单位米）
      - 刚性段：红色（段内采样，单位米）
    并且每隔 save_every 轮将当前回调结果保存至磁盘（PNG，且可选NPZ）。
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Convergence (live)")
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.05, 0.05])

    from pathlib import Path
    out_dir = Path(save_dir) if save_dir is not None else Path("callback_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    def cb(iter_num: int, z: Array, normE: float):
        if (iter_num % every) != 0:
            return

        # 清除先前的内容并重设坐标轴属性
        ax.cla()
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Convergence (live)")
        ax.view_init(elev=30, azim=30)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([-0.05, 0.08])

        x_nodes_list_bar, _, _ = unpack_z_bar_jax(z, M_list=params.M_list)

        # 柔段使用不透明蓝色
        flex_color = "blue"
        flex_alpha = 1.0
        # 刚段不透明度可随残差略变（可选）
        if normE > 1e-2:
            rigid_alpha = 0.9
        elif normE > 1e-3:
            rigid_alpha = 0.7
        elif normE > 1e-4:
            rigid_alpha = 0.5
        else:
            rigid_alpha = 0.4

        for i in range(len(params.flex)):
            # 柔性段：节点（bar->SI）
            p_flex_bar = x_nodes_list_bar[i][:, 0:3]
            p_flex_dim = p_flex_bar * scales.L_ref
            ax.plot(p_flex_dim[:, 0], p_flex_dim[:, 1], p_flex_dim[:, 2],
                    color=flex_color, alpha=flex_alpha, linewidth=1.8)

            # 刚性段：解析推进并采样
            rigidp = params.rigid[i]
            Lr = float(rigidp.length)
            if Lr <= 0.0 or n_samples_rigid <= 0:
                continue

            x_prox_dim = x_bar_to_dim(x_nodes_list_bar[i][-1], scales)

            f_total_dim, tau_total_dim = compute_external_wrench_total_rigid(
                x_proximal=x_prox_dim,
                rigid_length=Lr,
                gravity=params.gravity_rigid_list[i],
                magnetic_model=params.magnetic_model,
                magnet_params=params.magnet_params_list[i],
                coil_currents=params.coil_currents,
            )

            from pose_modules_nondim_jax.segments_nondim_jax import rigid_state_along_dim

            sigmas = np.linspace(0.0, Lr, int(n_samples_rigid) + 1)[1:]
            p_rigid = []
            for s in sigmas:
                x_s = rigid_state_along_dim(
                    x_prox_dim=x_prox_dim,
                    sigma=float(s),
                    rigid=rigidp,
                    f_ext_total_dim=f_total_dim,
                    tau_ext_total_dim=tau_total_dim,
                )
                p_rigid.append(np.asarray(x_s[0:3]))
            if p_rigid:
                pr = np.asarray(p_rigid)
                ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], "-o", color="r", markersize=2.0, linewidth=1.5, alpha=rigid_alpha)

        ax.set_title(f"Convergence (iter={iter_num}, ||E||={normE:.3e})")
        plt.tight_layout()
        plt.pause(0.01)

        # 间隔保存当前图像和可选中间解
        if save_every > 0 and (iter_num % save_every) == 0:
            png_path = out_dir / f"callback_iter_{iter_num:06d}.png"
            try:
                fig.savefig(png_path, dpi=150)
                print(f"[callback] saved {png_path}")
            except Exception as e:
                print(f"[callback] save png failed: {e}")
            if save_npz:
                npz_path = out_dir / f"callback_iter_{iter_num:06d}.npz"
                try:
                    np.savez(npz_path, z=np.asarray(z), iter=iter_num, normE=float(normE))
                except Exception as e:
                    print(f"[callback] save npz failed: {e}")

    return cb


def main():
    parser = build_argparser()
    args = parser.parse_args()

    cfg = load_config_file(args.config)

    def _get(name: str, default):
        # CLI overrides config
        v_cli = getattr(args, name, None)
        if v_cli is not None and (not isinstance(v_cli, bool) or v_cli is True or default is False):
            return v_cli
        return cfg.get(name, default)

    flex_lengths = _parse_csv_floats(_get("flex_lengths", args.flex_lengths))
    rigid_lengths = _parse_csv_floats(_get("rigid_lengths", args.rigid_lengths))
    M_list = _parse_csv_ints(_get("M_list", args.M_list))

    # Advancer model: if L_protrude is provided, infer the first flexible length L1.
    L_protrude = _get("L_protrude", args.L_protrude)
    L1_min = float(_get("L1_min", args.L1_min))

    if L_protrude is not None:
        # Segment count N is determined by rigid_lengths and M_list.
        if len(rigid_lengths) != len(M_list):
            raise ValueError(
                f"When using --L_protrude, segment count is determined by rigid_lengths and M_list. "
                f"Got len(rigid_lengths)={len(rigid_lengths)} vs len(M_list)={len(M_list)}."
            )
        flex_lengths = apply_advancer_protrude_length(
            flex_lengths_in=flex_lengths,
            rigid_lengths=rigid_lengths,
            L_protrude=float(L_protrude),
            L1_min=L1_min,
        )

    if not (len(flex_lengths) == len(rigid_lengths) == len(M_list)):
        raise ValueError(f"Segment count mismatch: len(flex_lengths)={len(flex_lengths)}, len(rigid_lengths)={len(rigid_lengths)}, len(M_list)={len(M_list)}")
    N = len(flex_lengths)

    flex_d_outer = _parse_csv_floats(_get("flex_d_outer", args.flex_d_outer))
    flex_E = _parse_csv_floats(_get("flex_E", args.flex_E))
    flex_G = _parse_csv_floats(_get("flex_G", args.flex_G))
    flex_rho = _parse_csv_floats(_get("flex_rho", args.flex_rho))

    rigid_d_outer = _parse_csv_floats(_get("rigid_d_outer", args.rigid_d_outer))
    rigid_rho = _parse_csv_floats(_get("rigid_rho", args.rigid_rho))

    p0_dim = _parse_vec3(_get("p0", args.p0))
    Q0 = _parse_quat(_get("Q0", args.Q0))
    axis_body = _parse_vec3(_get("axis_body", args.axis_body))

    enable_gravity = bool(cfg.get("enable_gravity", True)) or bool(args.enable_gravity)
    g_world = _parse_vec3(_get("g_world", args.g_world))

    enable_magnetics = bool(cfg.get("enable_magnetics", True)) or bool(args.enable_magnetics)
    calib_file = _get("calib_file", args.calib_file)
    actuation_table_pkl = _get("actuation_table_pkl", args.actuation_table_pkl)
    coil_currents_s = _get("coil_currents", args.coil_currents)
    coil_currents = None if coil_currents_s is None else jnp.array(_parse_csv_floats(coil_currents_s), dtype=jnp.float64)

    m_body_list_s = _get("m_body_list", args.m_body_list)
    m_body_list = None if m_body_list_s is None else _parse_list_vec3(m_body_list_s)

    total_L = float(sum(flex_lengths) + sum(rigid_lengths))  # current protruding total length
    # Scale choice uses segment 0 flexible material/geometry after broadcasting.
    flex_d0 = _repeat_or_validate(flex_d_outer, N, name="flex_d_outer")[0]
    flex_E0 = _repeat_or_validate(flex_E, N, name="flex_E")[0]
    flex_G0 = _repeat_or_validate(flex_G, N, name="flex_G")[0]
    scales = compute_scales_from_flex(L_ref=total_L, d_outer=float(flex_d0), E=float(flex_E0), G=float(flex_G0))

    params, meshes = build_solver_params(
        flex_lengths=flex_lengths,
        rigid_lengths=rigid_lengths,
        M_list=M_list,
        flex_d_outer=flex_d_outer,
        flex_E=flex_E,
        flex_G=flex_G,
        flex_rho=flex_rho,
        rigid_d_outer=rigid_d_outer,
        rigid_rho=rigid_rho,
        scales=scales,
        p0_dim=p0_dim,
        Q0=Q0,
        axis_body=axis_body,
        enable_gravity=enable_gravity,
        g_world=g_world,
        enable_magnetics=enable_magnetics,
        calib_file=calib_file,
        actuation_table_pkl=actuation_table_pkl,
        coil_currents=coil_currents,
        m_body_list=m_body_list,
    )

    z0_bar, _, _, _ = make_initial_guess_multi_bar_jax(
        flex_segs=list(params.flex),
        meshes=meshes,
        rigid_segs=list(params.rigid),
        scales=scales,
        p0_dim=p0_dim,
        Q0=Q0,
        axis_body=axis_body,
    )

    print("JAX devices:", jax.devices())
    solver = MultiSegmentEquilibriumSolverNondimJAX(params)

    # ===================== DIAG PATCH BEGIN =====================
    # 你希望检查 it=1 和 it=5001；注意：solve_lm 的 callback 会收到它自己打印的迭代号
    #（你的日志里 DIAG it=1 / 5001 已经验证这个编号是对齐的）
    debug_iters = {1, 5001}

    # 你关心的“具体块”：可按需改
    # 建议至少包含：平台期 state 最大的块（例如 seg0 interval1/2）以及段间连接 BV（seg1/seg2 的 interval0）
    watch_flex = [
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
    ]

    def _norm(x: Array) -> float:
        return float(jnp.linalg.norm(x))

    def _maxabs(x: Array) -> float:
        return float(jnp.max(jnp.abs(x))) if x.size > 0 else 0.0

    def _flex_block_index(seg: int, interval: int) -> int:
        # 全局 flex block idx = sum_{s<seg} M_list[s] + interval
        return int(sum(int(params.M_list[s]) for s in range(seg)) + int(interval))

    def _print_state_breakdown(vec13: Array, prefix: str = "") -> None:
        p = vec13[0:3]
        Q = vec13[3:7]
        f = vec13[7:10]
        tau = vec13[10:13]
        print(
            f"{prefix}state breakdown | "
            f"p={_norm(p):.3e} (max={_maxabs(p):.3e}), "
            f"Q={_norm(Q):.3e} (max={_maxabs(Q):.3e}), "
            f"f={_norm(f):.3e} (max={_maxabs(f):.3e}), "
            f"tau={_norm(tau):.3e} (max={_maxabs(tau):.3e})"
        )

    def _print_bv_breakdown(bv: Array, prefix: str = "") -> None:
        ln = int(bv.size)
        if ln == 0:
            print(f"{prefix}BV breakdown | (empty)")
            return

        if ln == 7:
            p = bv[0:3]
            Q = bv[3:7]
            print(
                f"{prefix}BV breakdown (len=7: p,Q) | "
                f"p={_norm(p):.3e} (max={_maxabs(p):.3e}), "
                f"Q={_norm(Q):.3e} (max={_maxabs(Q):.3e})"
            )
            return

        if ln == 13:
            print(f"{prefix}BV breakdown (len=13: p,Q,f,tau) | total={_norm(bv):.3e} (max={_maxabs(bv):.3e})")
            _print_state_breakdown(bv, prefix=prefix + "  ")
            return

        # 兜底：未知长度直接报数值
        print(f"{prefix}BV breakdown (len={ln}) | total={_norm(bv):.3e} (max={_maxabs(bv):.3e})")

    def _print_flex_block_details(E: Array, seg: int, interval: int, prefix: str = "") -> None:
        cs_len = int(params.cs_len)
        idx = _flex_block_index(seg, interval)
        off = int(params.flex_block_offsets[idx])
        ln = int(params.flex_block_lens[idx])
        blk = E[off:off + ln]

        C_S = blk[:cs_len]
        state = blk[cs_len:cs_len + 13]
        ks = blk[cs_len + 13:cs_len + 13 + 39]
        BV = blk[cs_len + 13 + 39:]

        # ks 分 3 个 stage
        ks0 = ks[0:13]
        ks1 = ks[13:26]
        ks2 = ks[26:39]

        print(f"{prefix}[FLEX-DETAIL] seg={seg} interval={interval} idx={idx} off={off} len={ln}")
        print(
            f"{prefix}  norms | total={_norm(blk):.3e} (max={_maxabs(blk):.3e}), "
            f"C_S={_norm(C_S):.3e}, state={_norm(state):.3e}, "
            f"ks={_norm(ks):.3e} (stages [{_norm(ks0):.3e}, {_norm(ks1):.3e}, {_norm(ks2):.3e}]), "
            f"BV={_norm(BV):.3e} (bv_len={int(BV.size)})"
        )
        _print_state_breakdown(state, prefix=prefix + "  ")
        _print_bv_breakdown(BV, prefix=prefix + "  ")

    def lm_callback(it: int, z_bar: Array, normE: float) -> None:
        # 只在你关心的迭代点打印
        if it not in debug_iters:
            return

        # 计算当前残差（会触发同步；但你只打印两次，所以成本可接受）
        E = solver.residual_jit(z_bar)

        print("\n" + "=" * 88)
        print(f"[DIAG] it={it}  ||E||={float(jnp.linalg.norm(E)):.6e}  (solver_normE={float(normE):.6e})")

        # 1) 原有总览（与你现在看到的一致）
        print_top_blocks_jax(
            E,
            params,
            topk_flex=12,
            topk_rigid=6,
            verbose=True,
            title_prefix=f"[DIAG it={it}] ",
        )

        # 2) 你指定的“重点块”做更细的 state / BV 分解
        print(f"[DIAG it={it}] Focused flex block decomposition:")
        for (seg, interval) in watch_flex:
            try:
                _print_flex_block_details(E, seg=int(seg), interval=int(interval), prefix=f"[DIAG it={it}] ")
            except Exception as ex:
                print(f"[DIAG it={it}]  (seg={seg}, interval={interval}) failed to parse: {ex}")

        print("=" * 88 + "\n")

    # ===================== DIAG PATCH END =====================

    cb = create_iteration_callback(params, scales, every=100, n_samples_rigid=10, save_every=100,
                                   save_dir="callback_snaps", save_npz=True)

    z_star, ok = solver.solve_lm(
        z0_bar,
        max_iter=int(_get("max_iter", args.max_iter)),
        tol=float(_get("tol", args.tol)),
        lm_damping=float(_get("lm_damping", args.lm_damping)),
        jac_method=str(_get("jac_method", args.jac_method)),
        callback=cb,
    )
    print("Converged:", ok)

    if args.plot or bool(cfg.get("plot", False)):
        plot_converged_pose_3d(z_star, params, scales, n_samples_rigid=8, show=True)


if __name__ == "__main__":
    main()


