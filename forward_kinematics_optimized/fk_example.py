
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
from basics_nondim import quat_normalize, quat_to_rotmat
from segments_nondim import FlexibleParams, RigidParams
from external_wrench_nondim import GravityRigid, MagneticModel, compute_external_wrench_total_rigid
from equilibrium_solver_nondim import SolverParams, MultiSegmentEquilibriumSolverNondimJAX
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
    # --- Layer 1 + External estimate A (optional) ---
    enable_magnetics: bool = False,
    calib_file: Optional[str] = None,
    actuation_table_pkl: Optional[str] = None,
    coil_currents: Optional[Array] = None,
    m_body_list: Optional[List[Array]] = None,
    # nominal pose used only for external scale estimation
    p0_dim: Optional[Array] = None,
    Q0: Optional[Array] = None,
    axis_body: Optional[Array] = None,
    flex_lengths: Optional[List[float]] = None,
    rigid_lengths: Optional[List[float]] = None,
) -> NondimScales:
    """Layer 1 scales (recommended first version).

    Internal scales (physics-consistent):
      - F_int_ref ~ median(diag(K_se))   [N]
      - M_int_ref ~ median(diag(K_bt)) / L_ref   [N*m]

    External estimate A (optional): estimate typical magnetic wrench at a
    nominal pose (based on p0/Q0/axis and segment lengths).

    Final scales:
      - F_ref = max(F_int_ref, F_ext_typ)
      - M_ref = max(M_int_ref, M_ext_typ)

    Notes:
      - This is nondimensionalization, NOT residual re-weighting.
      - If magnetics inputs are incomplete/unavailable, external terms default
        to 0.
    """
    # --- internal refs ---
    K_se, K_bt = build_k_matrices_circular_jax(d_outer=d_outer, E=E, G=G)
    diag_Kse = jnp.diag(K_se)
    diag_Kbt = jnp.diag(K_bt)
    F_int_ref = float(jnp.median(diag_Kse))
    M_int_ref = float(jnp.median(diag_Kbt) / jnp.maximum(float(L_ref), 1e-12))

    # --- external estimate A (typical wrench) ---
    F_ext_typ = 0.0
    M_ext_typ = 0.0
    if enable_magnetics:
        has_inputs = (calib_file is not None) and (coil_currents is not None) and (m_body_list is not None)
        has_pose = (p0_dim is not None) and (Q0 is not None) and (axis_body is not None)
        has_geom = (flex_lengths is not None) and (rigid_lengths is not None)
        if has_inputs and has_pose and has_geom and os.path.isfile(str(calib_file)):
            try:
                mas = MagneticActuationSystem(
                    calib_file=str(calib_file),
                    actuation_table_pkl=actuation_table_pkl,
                    dtype=jnp.float32,
                    enable_x64=True,
                )
                I = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(8,)
                p0 = jnp.asarray(p0_dim, dtype=jnp.float64).reshape(3,)
                q0 = quat_normalize(jnp.asarray(Q0, dtype=jnp.float64).reshape(4,))
                R0 = quat_to_rotmat(q0)
                a_body = jnp.asarray(axis_body, dtype=jnp.float64).reshape(3,)
                a_body = a_body / jnp.maximum(jnp.linalg.norm(a_body), 1e-12)
                a_world = R0 @ a_body

                cum = 0.0
                f_sum = 0.0
                tau_sum = 0.0
                for i, m_body in enumerate(m_body_list):
                    lf = float(flex_lengths[i])
                    lr = float(rigid_lengths[i])
                    s_mid = cum + lf + 0.5 * lr
                    p_i = p0 + a_world * s_mid
                    m_b = jnp.asarray(m_body, dtype=jnp.float64).reshape(3,)
                    m_w = R0 @ m_b
                    pose = p_i.reshape(3, 1)
                    m_vec = m_w.reshape(3, 1)
                    wrench = mas.magnetic_wrench(pose_list=pose, magnetic_moment=m_vec, currents_vector=I)
                    f_i = wrench[0:3, 0]
                    tau_i = wrench[3:6, 0]
                    f_sum = f_sum + float(jnp.linalg.norm(f_i))
                    tau_sum = tau_sum + float(jnp.linalg.norm(tau_i))
                    cum = cum + lf + lr

                F_ext_typ = float(f_sum)
                M_ext_typ = float(tau_sum)
            except Exception:
                F_ext_typ = 0.0
                M_ext_typ = 0.0

    eps = 1e-12
    F_ref = float(max(F_int_ref, F_ext_typ, eps))
    M_ref = float(max(M_int_ref, M_ext_typ, eps))
    return NondimScales(L_ref=float(L_ref), F_ref=F_ref, M_ref=M_ref)


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
    p.add_argument("--flex_lengths", type=str, default="0.03,0.03,0.03", help="Comma-separated flexible lengths [m].")
    p.add_argument("--rigid_lengths", type=str, default="0.003,0.003,0.003", help="Comma-separated rigid lengths [m].")
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
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--lm_damping", type=float, default=1e-3)
    p.add_argument("--jac_method", type=str, default="fwd", choices=["fwd", "rev"])
    p.add_argument("--plot", action="store_true", help="Plot converged pose (matplotlib).")
    return p


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    total_L = float(sum(flex_lengths) + sum(rigid_lengths))
    # Scale choice uses segment 0 flexible material/geometry after broadcasting.
    flex_d0 = _repeat_or_validate(flex_d_outer, N, name="flex_d_outer")[0]
    flex_E0 = _repeat_or_validate(flex_E, N, name="flex_E")[0]
    flex_G0 = _repeat_or_validate(flex_G, N, name="flex_G")[0]
    scales = compute_scales_from_flex(
        L_ref=total_L,
        d_outer=float(flex_d0),
        E=float(flex_E0),
        G=float(flex_G0),
        enable_magnetics=enable_magnetics,
        calib_file=calib_file,
        actuation_table_pkl=actuation_table_pkl,
        coil_currents=coil_currents,
        m_body_list=m_body_list,
        p0_dim=p0_dim,
        Q0=Q0,
        axis_body=axis_body,
        flex_lengths=flex_lengths,
        rigid_lengths=rigid_lengths,
    )

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

    z_star, ok = solver.solve_lm(
        z0_bar,
        max_iter=int(_get("max_iter", args.max_iter)),
        tol=float(_get("tol", args.tol)),
        lm_damping=float(_get("lm_damping", args.lm_damping)),
        jac_method=str(_get("jac_method", args.jac_method)),
    )
    print("Converged:", ok)

    if args.plot or bool(cfg.get("plot", False)):
        plot_converged_pose_3d(z_star, params, scales, n_samples_rigid=8, show=True)


if __name__ == "__main__":
    main()


