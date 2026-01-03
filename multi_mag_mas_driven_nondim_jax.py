# multi_mag_mas_driven_nondim_jax.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# --- Only import from the requested scripts ---
from pose_modules_nondim_jax.nondim_jax import NondimScales
from pose_modules_nondim_jax.segments_nondim_jax import FlexibleParams, RigidParams
from pose_modules_nondim_jax.external_wrench_nondim_jax import GravityRigid, MagneticModel
from pose_modules_nondim_jax.equilibrium_solver_nondim_jax import SolverParams, MultiSegmentEquilibriumSolverNondimJAX, audit_last_rigid_force_identity
from pose_modules_nondim_jax.utils_nondim_jax import (
    make_initial_guess_multi_bar_jax,
    unpack_z_bar_jax,
    build_k_matrices_for_pdms_jax,
    build_gravity_line_density_for_pdms_jax,
    build_gravity_rigid_for_ndfeb_jax,
)
from pose_modules_nondim_jax.nondim_jax import x_bar_to_dim
from pose_modules_nondim_jax.external_wrench_nondim_jax import compute_external_wrench_total_rigid
from pose_modules_nondim_jax.rod_mesh_nondim_jax import UniformMesh, build_uniform_mesh
from pose_modules_nondim_jax.basics_nondim_jax import quat_to_rotmat, quat_normalize

# Optional: Supiee MAS (still within your allowed list)
from pose_modules_nondim_jax.mas_nondim_jax import MagneticActuationSystem

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


def rigid_var_offset_in_z(params, seg: int) -> int:
    """
    Return the starting index of xR_var_bar (13) for segment seg in z_bar layout:
      for each seg i:
        nodes: (M+1)*13
        k:     M*39
        rigid: 13
    """
    idx = 0
    N = len(params.flex)
    for i in range(N):
        M = int(params.M_list[i])
        idx += (M + 1) * 13
        idx += M * 39
        if i == seg:
            return idx
        idx += 13
    raise ValueError(f"seg={seg} out of range")

def flex_end_offset_in_z(params, seg: int) -> int:
    """
    Return the starting index of the last node state x_nodes[seg][-1] (13) in z_bar.
    """
    idx = 0
    N = len(params.flex)
    for i in range(N):
        M = int(params.M_list[i])
        # nodes block
        if i == seg:
            # last node is (M)*13 after nodes start
            return idx + M * 13
        idx += (M + 1) * 13
        idx += M * 39
        idx += 13
    raise ValueError(f"seg={seg} out of range")

def print_rigid_state_fields(z_bar, params, seg: int, *, label: str = "", raise_on_fail: bool = False):
    z_bar = jnp.asarray(z_bar, dtype=jnp.float64)

    offR = rigid_var_offset_in_z(params, seg)
    xR = z_bar[offR:offR + 13]

    p = xR[0:3]
    q = xR[3:7]
    f = xR[7:10]
    tau = xR[10:13]

    finite_ok = bool(jnp.all(jnp.isfinite(xR)))
    qnorm = float(jnp.dot(q, q))
    pnorm = float(jnp.linalg.norm(p))
    fnorm = float(jnp.linalg.norm(f))
    taunorm = float(jnp.linalg.norm(tau))

    print(f"{label}[CHECK rigid xR_var] seg={seg} off={offR}")
    print(f"{label}  finite={finite_ok}  ||p||={pnorm:.6g}  ||f||={fnorm:.6g}  ||tau||={taunorm:.6g}  |q|^2={qnorm:.6g}")
    print(f"{label}  p={np.asarray(jax.device_get(p))}")
    print(f"{label}  q={np.asarray(jax.device_get(q))}")
    print(f"{label}  f={np.asarray(jax.device_get(f))}")
    print(f"{label}  tau={np.asarray(jax.device_get(tau))}")

    # 断言：有限值
    if raise_on_fail and not finite_ok:
        raise AssertionError(f"Non-finite xR_var detected at seg={seg}")

    # 断言：四元数范数不应离谱（初值/收敛后一般接近1）
    if raise_on_fail and not (0.5 < qnorm < 1.5):
        raise AssertionError(f"Quaternion norm^2 abnormal at seg={seg}: {qnorm}")

def check_last_rigid_bv_sensitivity(solver, z_bar, *, seg: int, eps: float = 1e-6, label: str = ""):
    """
    Verify residual depends on xR_var's tip force/moment components.
    For last seg, BV residual should include [f_end, tau_end] (bar), so perturbing them should change BV ~ O(eps).
    """
    p = solver.params
    z_bar = jnp.asarray(z_bar, dtype=jnp.float64)

    # baseline residual
    E0 = solver.residual_jit(z_bar)

    # locate rigid block residual slice
    off_blk = int(p.rigid_block_offsets[seg])
    ln_blk  = int(p.rigid_block_lens[seg])
    R0 = E0[off_blk:off_blk + ln_blk]
    BV0 = R0[14:]  # [f_end(3), tau_end(3)] for last seg; empty for non-last

    offR = rigid_var_offset_in_z(p, seg)

    # perturb f_end_var (3)
    z1 = z_bar.at[offR + 7: offR + 10].add(eps)
    E1 = solver.residual_jit(z1)
    R1 = E1[off_blk:off_blk + ln_blk]
    BV1 = R1[14:]

    # perturb tau_end_var (3)
    z2 = z_bar.at[offR + 10: offR + 13].add(eps)
    E2 = solver.residual_jit(z2)
    R2 = E2[off_blk:off_blk + ln_blk]
    BV2 = R2[14:]

    dBV_f = float(jnp.linalg.norm(BV1 - BV0)) if BV0.size else 0.0
    dBV_t = float(jnp.linalg.norm(BV2 - BV0)) if BV0.size else 0.0

    print(f"{label}[SENS BV] seg={seg} ln_blk={ln_blk} bv_len={int(BV0.size)} eps={eps:g}")
    print(f"{label}  ||BV(z+eps_f) - BV(z)|| = {dBV_f:.6g}   (expected ~ O(eps))")
    print(f"{label}  ||BV(z+eps_tau) - BV(z)|| = {dBV_t:.6g} (expected ~ O(eps))")

    # 断言：如果是末段，BV 长度应该是 6，且敏感度不应接近 0
    if seg == len(p.rigid) - 1:
        if int(BV0.size) != 6:
            raise AssertionError(f"Last seg BV length expected 6, got {int(BV0.size)}")
        if dBV_f < 1e-12 or dBV_t < 1e-12:
            raise AssertionError("BV appears insensitive to xR_var tip components; likely slicing/layout bug or gradient cut.")


# ----------------------------- Scales (nondimensionalization) -----------------------------

def compute_scales_pdms(
    *,
    d_outer: float,
    L_ref: float,
) -> NondimScales:
    """
    A pragmatic scale choice:
      - L_ref: choose total catheter length (or any representative length)
      - F_ref: choose axial stiffness scale ~ E*A  (max diag of K_se)
      - M_ref: F_ref * L_ref
    """
    K_se, _ = build_k_matrices_for_pdms_jax(d_outer)
    F_ref = float(jnp.max(jnp.diag(K_se)))
    M_ref = float(F_ref * float(L_ref))
    return NondimScales(L_ref=float(L_ref), F_ref=float(F_ref), M_ref=float(M_ref))


# ----------------------------- Magnetic model adapter (Rigid total wrench) -----------------------------

class SupieeRigidMagneticModelJAX(MagneticModel):
    """
    Adapter to satisfy external_wrench_nondim_jax.MagneticModel protocol:
        compute_wrench_world(p_world, R_world_from_body, magnet_params, coil_currents) -> (f_world, tau_world)

    Uses MagneticActuationSystem.magnetic_wrench() from mas_nondim_jax.
    """
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


# ----------------------------- Build SolverParams (pure JAX) -----------------------------

def build_solver_params(
    *,
    d_outer: float,
    flex_lengths: List[float],
    rigid_lengths: List[float],
    M_list: List[int],
    scales: NondimScales,
    enable_gravity: bool = True,
    enable_magnetics: bool = True,
    calib_file: Optional[str] = None,
    actuation_table_pkl: Optional[str] = None,
    coil_currents: Optional[Array] = None,
    m_body_list: Optional[List[Array]] = None,
    # 用户可选：设置近端初始位置
    # 传 SI (米) 用 p0_dim；或直接传无量纲的 p0_bar
    p0_dim: Optional[Array] = None,
    p0_bar: Optional[Array] = None,
) -> Tuple[SolverParams, List[UniformMesh]]:
    """
    Create SolverParams without numpy solver and without rod_mesh.py.
    """
    assert len(flex_lengths) == len(rigid_lengths) == len(M_list)
    N = len(flex_lengths)

    # --- Meshes ---
    meshes: List[UniformMesh] = [build_uniform_mesh(flex_lengths[i], M_list[i]) for i in range(N)]

    # --- Flexible parameters (use PDMS K matrices; v_star/u_star per your convention) ---
    K_se, K_bt = build_k_matrices_for_pdms_jax(d_outer)
    Kse_inv = jnp.linalg.inv(jnp.asarray(K_se, dtype=jnp.float64))
    Kbt_inv = jnp.linalg.inv(jnp.asarray(K_bt, dtype=jnp.float64))

    v_star = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
    u_star = jnp.zeros((3,), dtype=jnp.float64)

    flex_params: List[FlexibleParams] = []
    for i in range(N):
        flex_params.append(
            FlexibleParams(
                length=float(flex_lengths[i]),
                Kse_inv=Kse_inv,
                Kbt_inv=Kbt_inv,
                v_star=v_star,
                u_star=u_star,
            )
        )

    # --- Rigid parameters (v_star is direction of centerline in body frame) ---
    rigid_params: List[RigidParams] = []
    for i in range(N):
        rigid_params.append(
            RigidParams(
                length=float(rigid_lengths[i]),
                v_star=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
            )
        )

    # --- Proximal BC target ---
    # 优先级：p0_bar（直接使用）> p0_dim（用 L_ref 归一化）> 默认 0
    if p0_bar is not None:
        p0_bar = jnp.asarray(p0_bar, dtype=jnp.float64).reshape(3,)
    elif p0_dim is not None:
        p0_bar = jnp.asarray(p0_dim, dtype=jnp.float64).reshape(3,) / float(scales.L_ref)
    else:
        p0_bar = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
    Q0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

    # --- Flexible distributed loads: gravity as constant world line density (SI) ---
    if enable_gravity:
        g_line = build_gravity_line_density_for_pdms_jax(d_outer).force_world()  # (3,) N/m
    else:
        g_line = jnp.zeros((3,), dtype=jnp.float64)

    flex_f_line_world_list = tuple(g_line for _ in range(N))
    flex_tau_line_world_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))

    # --- Rigid lumped loads: gravity + magnetics computed inside solver residual ---
    gravity_rigid_list: List[Optional[GravityRigid]] = []
    for i in range(N):
        if enable_gravity:
            gravity_rigid_list.append(build_gravity_rigid_for_ndfeb_jax(d_outer, float(rigid_lengths[i])))
        else:
            gravity_rigid_list.append(None)

    # Additional user-provided totals (kept as zeros unless you need them)
    f_ext_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))
    tau_ext_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))

    # --- Magnetics setup ---
    magnetic_model = None
    magnet_params_list: List[Optional[Dict]] = [None for _ in range(N)]
    coil_currents_out = None

    if enable_magnetics:
        if calib_file is None:
            raise ValueError("enable_magnetics=True but calib_file is None.")
        if coil_currents is None:
            raise ValueError("enable_magnetics=True but coil_currents is None.")
        if m_body_list is None or len(m_body_list) != N:
            raise ValueError("enable_magnetics=True requires m_body_list with length == N.")

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

    # --- Residual layout (must match equilibrium_solver_nondim_jax.residual_bar stacking) ---
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
        flex_f_line_world_list=flex_f_line_world_list,
        flex_tau_line_world_list=flex_tau_line_world_list,
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


def sanity_check_geometry_and_pack(
    *,
    scales,
    flex_lengths,
    rigid_lengths,
    meshes,
    params,
    z0_bar,
):
    import jax.numpy as jnp

    print("\n[SANITY] --- segment lengths (SI meters) ---")
    print("  flex_lengths (m):", [float(x) for x in flex_lengths])
    print("  rigid_lengths(m):", [float(x) for x in rigid_lengths])
    total_L = float(sum([float(x) for x in flex_lengths]) + sum([float(x) for x in rigid_lengths]))
    print("  total length (m):", total_L)
    print("  scales.L_ref (m):", float(scales.L_ref))
    if abs(total_L - float(scales.L_ref)) / max(1e-12, float(scales.L_ref)) > 0.2:
        print("  [WARN] total_L differs from L_ref by >20%. This can be OK, but often indicates unit mismatch.")

    print("\n[SANITY] --- mesh endpoints ---")
    for i, mesh in enumerate(meshes):
        Lseg = float(flex_lengths[i])
        sigma_end = float(mesh.sigma_nodes[-1])
        sbar_end = float(mesh.sbar_nodes[-1])
        hbar_sum = float(jnp.sum(mesh.hbar_list))
        print(f"  seg={i}: flex_length={Lseg:.6g}  sigma_end={sigma_end:.6g}  sbar_end={sbar_end:.6g}  sum(hbar)={hbar_sum:.6g}")
        if abs(sigma_end - Lseg) / max(1e-12, Lseg) > 1e-6:
            print("    [FAIL] sigma_nodes[-1] != flex_length. Mesh length mismatch (unit/constructor bug).")
        if abs(sbar_end - 1.0) > 1e-12:
            print("    [FAIL] sbar_nodes[-1] != 1.0. Nondim mesh should end at 1.")
        if abs(hbar_sum - 1.0) > 1e-12:
            print("    [FAIL] sum(hbar_list) != 1.0. Step lengths should sum to 1.")

    # ---- initial guess: compare expected z accumulation vs actual ----
    print("\n[SANITY] --- initial guess end positions (SI meters) ---")
    x_nodes_list_bar, k_list_bar, x_rigid_list_bar = unpack_z_bar_jax(z0_bar, M_list=params.M_list)

    z_expected = 0.0
    for i in range(len(flex_lengths)):
        Lf = float(flex_lengths[i])
        Lr = float(rigid_lengths[i])

        # flex end from nodes (bar -> SI meters)
        p_flex_end_bar = x_nodes_list_bar[i][-1, 0:3]
        p_flex_end_dim = p_flex_end_bar * float(scales.L_ref)

        # rigid end var (bar -> SI meters)
        p_rigid_end_bar = x_rigid_list_bar[i][0:3]
        p_rigid_end_dim = p_rigid_end_bar * float(scales.L_ref)

        z_expected_flex_end = z_expected + Lf
        z_expected_rigid_end = z_expected + Lf + Lr

        print(f"  seg={i}: flex_end.z={float(p_flex_end_dim[2]):.9g}  (expected ~{z_expected_flex_end:.9g})   "
              f"rigid_end.z={float(p_rigid_end_dim[2]):.9g} (expected ~{z_expected_rigid_end:.9g})")

        if abs(float(p_flex_end_dim[2]) - z_expected_flex_end) > 0.05 * max(1e-6, Lf):
            print("    [FAIL] flex_end.z far from expected. Likely length unit error OR pack/unpack mismatch OR dp scaling error.")
        if abs(float(p_rigid_end_dim[2]) - z_expected_rigid_end) > 0.05 * max(1e-6, Lr + Lf):
            print("    [FAIL] rigid_end.z far from expected. Likely length unit error OR rigid propagation/packing mismatch.")

        z_expected += (Lf + Lr)

    # ---- check K-array shapes (pack/unpack consistency) ----
    print("\n[SANITY] --- k-array shapes after unpack ---")
    for i, k_flat in enumerate(k_list_bar):
        print(f"  seg={i}: k_flat shape={tuple(k_flat.shape)}  (expected (M,39) with M={int(params.M_list[i])})")
        if k_flat.shape[0] != int(params.M_list[i]) or k_flat.shape[1] != 39:
            print("    [FAIL] k_flat shape mismatch. pack/unpack layout is inconsistent with solver assumptions.")

    print("\n[SANITY] done.\n")


def sanity_check_load_magnitudes(params):
    # flex line load density should be about rho*A*9.81 (N/m)
    for i, f_line in enumerate(params.flex_f_line_world_list):
        f_line = jnp.asarray(f_line)
        print(f"[LOAD] seg={i} ||f_line_world|| (N/m) =", float(jnp.linalg.norm(f_line)))

    # rigid gravity total should be about m*9.81 (N)
    for i, g in enumerate(params.gravity_rigid_list):
        if g is None:
            print(f"[LOAD] seg={i} rigid gravity: None")
        else:
            f = g.force_world()
            print(f"[LOAD] seg={i} ||F_g|| (N) =", float(jnp.linalg.norm(f)))


def plot_converged_pose_3d(
    z_bar: Array,
    params: SolverParams,
    scales: NondimScales,
    *,
    n_samples_rigid: int = 12,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    绘制收敛后的三维姿态（柔段节点 + 刚段采样点），单位为米。

    - 柔段：直接使用节点位置（bar -> SI）。
    - 刚段：基于解析 rigid_state_along 在段内均匀采样（SI）。
    - 外载：使用 compute_external_wrench_total_rigid 在刚段近端计算总合力/力矩（SI）。
    """
    # 解包
    x_nodes_list_bar, _, _ = unpack_z_bar_jax(z_bar, M_list=params.M_list)

    N = len(params.flex)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    any_plotted = False

    for i in range(N):
        # 柔段节点（bar -> SI）
        x_nodes_bar = jnp.asarray(x_nodes_list_bar[i])  # (M+1, 13)
        p_flex_m = np.asarray(x_nodes_bar[:, 0:3] * float(scales.L_ref))

        # --- quick sanity: first node of seg0 should be close to base position ---
        if i == 0:
            print("[PLOT DEBUG] seg0 node0 raw x(13) =", np.asarray(x_nodes_bar[0]))
            print("[PLOT DEBUG] seg0 node0 p_bar =", np.asarray(x_nodes_bar[0, 0:3]))
            print("[PLOT DEBUG] seg0 node0 q_bar =", np.asarray(x_nodes_bar[0, 3:7]))

            p0 = np.asarray(x_nodes_bar[0, 0:3]) * float(scales.L_ref)
            q0 = np.asarray(x_nodes_bar[0, 3:7])
            print("[PLOT DEBUG] seg0 node0 p(m) =", p0, " |q|^2 =", float(np.dot(q0, q0)))

        # compare unpacked last node vs manual slice from z_bar for seg=2
        if i == 2:
            # manual compute nodes start for seg=2:
            idx = 0
            for s in range(i):
                M = int(params.M_list[s])
                idx += (M + 1) * 13 + M * 39 + 13
            M = int(params.M_list[i])
            nodes_flat = jnp.asarray(z_bar[idx: idx + (M + 1) * 13])
            x_nodes_manual = nodes_flat.reshape((M + 1, 13))
            diff = np.linalg.norm(np.asarray(x_nodes_manual[-1] - x_nodes_bar[-1]))
            print("[PLOT DEBUG] seg2 last-node unpack-vs-manual ||diff|| =", diff)

        if p_flex_m.size:
            ax.plot(p_flex_m[:, 0], p_flex_m[:, 1], p_flex_m[:, 2], "-o", color="b", markersize=2.5, linewidth=1.5)
            any_plotted = True

        # 刚段采样
        rigidp = params.rigid[i]
        Lr = float(rigidp.length)
        if Lr <= 0.0 or n_samples_rigid <= 0:
            continue

        # 刚段近端的 SI 状态（柔段末端）
        from pose_modules_nondim_jax.nondim_jax import x_bar_to_dim
        from pose_modules_nondim_jax.segments_nondim_jax import rigid_state_along_dim
        from pose_modules_nondim_jax.external_wrench_nondim_jax import compute_external_wrench_total_rigid

        x_prox_dim = x_bar_to_dim(x_nodes_bar[-1], scales)

        # 计算刚段总外载（SI，关于近端）
        f_total_dim, tau_total_dim = compute_external_wrench_total_rigid(
            x_proximal=x_prox_dim,
            rigid_length=Lr,
            gravity=params.gravity_rigid_list[i],
            magnetic_model=params.magnetic_model,
            magnet_params=params.magnet_params_list[i],
            coil_currents=params.coil_currents,
        )

        # 沿刚段采样位置（排除近端以避免重复点）
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
            ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], "-o", color="r", markersize=2.5, linewidth=1.5)
            any_plotted = True

    if not any_plotted:
        print("[plot] No points to plot.")
        plt.close(fig)
        return

    # 限制轴范围
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.05, 0.05])

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Converged catheter pose (meters)")
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------- Iteration callback for live plotting -----------------------------

def create_iteration_callback(
    params: SolverParams,
    scales: NondimScales,
    *,
    every: int = 10,
    n_samples_rigid: int = 10,
):
    """
    在 LM 迭代过程中周期性（every 步）绘制当前姿态：
      - 柔性段：蓝色（节点，单位米）
      - 刚性段：红色（段内采样，单位米）
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

    def cb(iter_num: int, z: Array, normE: float):
        if (iter_num % every) != 0:
            return

        # 清除先前的内容并重设坐标轴属性
        ax.cla()
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Convergence (live)")
        ax.view_init(elev=30, azim=-60)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([-0.05, 0.08])

        x_nodes_list_bar, _, _ = unpack_z_bar_jax(z, M_list=params.M_list)

        # 透明度映射
        if normE > 1e-2:
            alpha = 0.9
        elif normE > 1e-3:
            alpha = 0.6
        elif normE > 1e-4:
            alpha = 0.3
        else:
            alpha = 0.15

        for i in range(len(params.flex)):
            # 柔性段：节点（bar->SI）
            p_flex_bar = x_nodes_list_bar[i][:, 0:3]
            p_flex_dim = p_flex_bar * scales.L_ref
            ax.plot(p_flex_dim[:, 0], p_flex_dim[:, 1], p_flex_dim[:, 2],
                    color="b", alpha=alpha, linewidth=1.5)

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
                ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], "-o", color="r", markersize=2.0, linewidth=1.5, alpha=alpha)

        ax.set_title(f"Convergence (iter={iter_num}, ||E||={normE:.3e})")
        plt.tight_layout()
        plt.pause(0.01)

    return cb

# ----------------------------- Main -----------------------------

def main():
    # ---- Basic run configuration (edit to match your experiment) ----
    d_outer = 0.0015  # [m]

    # Three flex + three rigid example (edit)
    flex_lengths = [0.03, 0.03, 0.03]       # [m]
    rigid_lengths = [0.003, 0.003, 0.003]   # [m]
    M_list = [5, 5, 5]                   # intervals per flex

    total_L = sum(flex_lengths) + sum(rigid_lengths)
    scales = compute_scales_pdms(d_outer=d_outer, L_ref=total_L)

    # ---- Set proximal initial position in SI (meters) ----
    p0_dim = jnp.array([0.0, 0.0, -0.0], dtype=jnp.float64)

    # ---- Magnetics (example) ----
    enable_magnetics = True
    calib_file = "/path/to/your/calibration.json"  # TODO: set
    actuation_table_pkl = None                     # or set explicit pkl
    # coil_currents = jnp.zeros((8,), dtype=jnp.float64)  # TODO: set your 8-coil currents
    coil_currents = jnp.array([3.3832,24.4370,-30.7323,-2.5516,-20.1070,44.8393,23.1883,-35.1559], dtype=jnp.float64)
    # coil_currents = jnp.array([5, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float64)

    # Example magnet moments in BODY frame for each rigid (A·m^2) - TODO: set real
    m_mag = 0.005301  # A·m^2
    m_body_list = [
        jnp.array([m_mag, 0.0, 0.0], dtype=jnp.float64),
        jnp.array([0.0, 0.0, m_mag], dtype=jnp.float64),
        jnp.array([0.0, 0.0, -m_mag], dtype=jnp.float64),
    ]

    # ---- Gravity on (per your statement) ----
    enable_gravity = True

    # ---- Build params and initial guess ----
    params, meshes = build_solver_params(
        d_outer=d_outer,
        flex_lengths=flex_lengths,
        rigid_lengths=rigid_lengths,
        M_list=M_list,
        scales=scales,
        enable_gravity=enable_gravity,
        enable_magnetics=enable_magnetics,
        calib_file=calib_file if enable_magnetics else None,
        actuation_table_pkl=actuation_table_pkl,
        coil_currents=coil_currents if enable_magnetics else None,
        m_body_list=m_body_list if enable_magnetics else None,
        p0_dim=p0_dim,
    )

    # Initial guess generator expects mesh.sigma_nodes, seg.length, rigid.length
    # We pass "flex-like" and "rigid-like" objects by reusing params.flex/params.rigid
    z0_bar, _, _, _ = make_initial_guess_multi_bar_jax(
        flex_segs=list(params.flex),
        meshes=meshes,
        rigid_segs=list(params.rigid),
        scales=scales,
    )

    sanity_check_geometry_and_pack(
        scales=scales,
        flex_lengths=flex_lengths,
        rigid_lengths=rigid_lengths,
        meshes=meshes,
        params=params,
        z0_bar=z0_bar,
    )


    # sanity_check_load_magnitudes(params)

    # ---- Solve with JAX AD Jacobian ----
    print("JAX devices:", jax.devices())
    solver = MultiSegmentEquilibriumSolverNondimJAX(params)

    # last = len(params.rigid) - 1
    # print_rigid_state_fields(z0_bar, params, last, label="[preLM] ", raise_on_fail=True)

    # 敏感度检查很关键
    # check_last_rigid_bv_sensitivity(solver, z0_bar, seg=last, eps=1e-6, label="[preLM] ")

    # def cb(it, z, normE):
    #     if it in (1, 2, 3, 4, 5) or (it % 5 == 0):
    #         solver.print_top_blocks(z, topk_flex=9, topk_rigid=3, verbose=True,
    #                                 title_prefix=f"[it={it}] ")

    # def cb(it, z, normE):
    #     # 建议：前几步 + 每隔若干步打印一次，避免拖慢
    #     if it in (0, 1, 2, 3, 5, 10, 20) or (it % 5 == 0):
    #         audit_last_rigid_force_identity(z, solver.params, it=it, seg_idx=None, print_vectors=False)
    #         solver.print_top_blocks(z, it=it, topk=10, verbose=True)
    #
    # def cb(it, z, normE):
    #     if it in (0, 1, 2, 5, 10) or (it % 20 == 0):
    #         print_rigid_state_fields(z, params, last, label=f"[it={it}] ", raise_on_fail=False)
    #         check_last_rigid_bv_sensitivity(solver, z, seg=last, eps=1e-6, label=f"[it={it}] ")

    # 可视化收敛过程：每 10 步绘制一次
    cb = create_iteration_callback(params, scales, every=100, n_samples_rigid=10)

    z_star, ok = solver.solve_lm(
        z0_bar,
        max_iter=600000,
        tol=1e-5,
        lm_damping=1e-3,
        jac_method="fwd",  # "rev" is sometimes better for very large z
        callback=cb,
    )
    print("Converged:", ok)

    # ---- Basic post: extract node positions (meters) ----
    x_nodes_list_bar, _, x_rigid_list_bar = unpack_z_bar_jax(z_star, M_list=params.M_list)

    for i in range(len(params.flex)):
        x_prox_dim = x_bar_to_dim(x_nodes_list_bar[i][-1], scales)

        f_total_dim, tau_total_dim = compute_external_wrench_total_rigid(
            x_proximal=x_prox_dim,
            rigid_length=float(params.rigid[i].length),
            gravity=params.gravity_rigid_list[i],
            magnetic_model=params.magnetic_model,
            magnet_params=params.magnet_params_list[i],
            coil_currents=params.coil_currents,
        )

        print(f"[LOAD CHECK] seg={i}")
        print("  p_prox(m) =", np.asarray(x_prox_dim[0:3]))
        print("  f_total(N)   =", np.asarray(f_total_dim), "||·||=", float(np.linalg.norm(np.asarray(f_total_dim))))
        print("  tau_total(Nm)=", np.asarray(tau_total_dim), "||·||=", float(np.linalg.norm(np.asarray(tau_total_dim))))

    for i in range(len(x_nodes_list_bar)):
        p_flex_bar = x_nodes_list_bar[i][:, 0:3]
        p_flex_dim = p_flex_bar * scales.L_ref
        p_rigid_end_dim = x_rigid_list_bar[i][0:3] * scales.L_ref
        print(f"[seg {i}] flex end (m):", p_flex_dim[-1])
        print(f"[seg {i}] rigid end (m):", p_rigid_end_dim)

    # If you want, you can add your own matplotlib plot here (not required for solver).
    try:
        plot_converged_pose_3d(z_star, params, scales, n_samples_rigid=5, save_path=None, show=True)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")


if __name__ == "__main__":
    main()
