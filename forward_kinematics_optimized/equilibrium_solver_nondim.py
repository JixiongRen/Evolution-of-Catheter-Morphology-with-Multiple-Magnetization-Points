# equilibrium_solver_nondim.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from nondim import NondimScales, x_bar_to_dim, rhs_dim_to_rhs_bar_dsbar
from segments_nondim import (
    FlexibleParams,
    RigidParams,
    cosserat_rhs_dim,
    interval_residual_gl6_bar,
    rigid_state_along_dim,
)

from external_wrench_nondim import (
    GravityRigid,
    MagneticModel,
    compute_external_wrench_total_rigid,
)

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


def with_coil_currents(params: "SolverParams", coil_currents: Array | None) -> "SolverParams":
    """Return a new :class:`SolverParams` with updated ``coil_currents``.

    Motivation
    ----------
    In the current FK implementation, the equilibrium residual is evaluated as
    ``E(z, params)`` where the coil currents are stored inside ``params``.

    For implicit differentiation / gradient-based IK, we need an explicit
    interface ``E(z, I)`` so that JAX can compute ``dE/dI``. This helper
    enables that with minimal, well-scoped changes.

    Notes
    -----
    - ``SolverParams`` is registered as a PyTree (see ``tree_flatten``).
      The coil currents are included in the children tuple, so replacing it
      keeps the structure compatible with JAX transforms.
    - This function is *pure*: it does not mutate ``params``.
    """
    children, aux = params.tree_flatten()
    # children layout ends with coil_currents
    *rest, _old_I = children
    new_children = tuple(rest) + (coil_currents,)
    return SolverParams.tree_unflatten(aux, new_children)


def residual_bar_zI(z_bar: Array, coil_currents: Array, params_static: "SolverParams") -> Array:
    """Explicit residual interface ``E(z, I)``.

    Args:
        z_bar: packed equilibrium state (nondimensional), shape (n_z,)
        coil_currents: coil currents, shape (8,)
        params_static: a ``SolverParams`` instance whose fields (except
            ``coil_currents``) define the static configuration.

    Returns:
        Residual vector ``E`` (nondimensional), shape (n_E,)
    """
    p = with_coil_currents(params_static, jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1,))
    return residual_bar(z_bar, p)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SolverParams:
    flex: Tuple[FlexibleParams, ...]
    rigid: Tuple[RigidParams, ...]
    sbar_nodes: Tuple[Array, ...]
    hbar_list: Tuple[Array, ...]
    M_list: Tuple[int, ...]

    p0_bar: Array
    Q0: Array

    # rigid lumped totals (world) about proximal
    f_ext_list: Tuple[Array, ...]
    tau_ext_list: Tuple[Array, ...]

    # flexible distributed loads in WORLD frame (SI)
    #   f_line_world: N/m
    #   tau_line_world: (N*m)/m
    flex_f_line_world_list: Tuple[Array, ...]
    flex_tau_line_world_list: Tuple[Array, ...]

    gravity_rigid_list: Tuple[Optional[GravityRigid], ...]
    magnet_params_list: Tuple[Optional[Dict], ...]
    coil_currents: Optional[Array]
    magnetic_model: Optional[MagneticModel]

    scales: NondimScales

    # residual layout (must match numpy assembly order)
    cs_len: int
    flex_block_offsets: Tuple[int, ...]
    flex_block_lens: Tuple[int, ...]
    rigid_block_offsets: Tuple[int, ...]
    rigid_block_lens: Tuple[int, ...]
    total_E_len: int

    def tree_flatten(self):
        children = (
            self.flex, self.rigid,
            self.sbar_nodes, self.hbar_list,
            self.p0_bar, self.Q0,
            self.f_ext_list, self.tau_ext_list,
            self.flex_f_line_world_list, self.flex_tau_line_world_list,
            self.gravity_rigid_list, self.magnet_params_list,
            self.coil_currents,
        )
        aux = dict(
            M_list=self.M_list,
            magnetic_model=self.magnetic_model,
            scales=self.scales,
            cs_len=self.cs_len,
            flex_block_offsets=self.flex_block_offsets,
            flex_block_lens=self.flex_block_lens,
            rigid_block_offsets=self.rigid_block_offsets,
            rigid_block_lens=self.rigid_block_lens,
            total_E_len=self.total_E_len,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            flex, rigid, sbar_nodes, hbar_list, p0_bar, Q0,
            f_ext_list, tau_ext_list,
            flex_f_line_world_list, flex_tau_line_world_list,
            gravity_rigid_list, magnet_params_list,
            coil_currents
        ) = children
        return cls(
            flex=tuple(flex),
            rigid=tuple(rigid),
            sbar_nodes=tuple(sbar_nodes),
            hbar_list=tuple(hbar_list),
            M_list=tuple(aux["M_list"]),
            p0_bar=p0_bar,
            Q0=Q0,
            f_ext_list=tuple(f_ext_list),
            tau_ext_list=tuple(tau_ext_list),
            flex_f_line_world_list=tuple(flex_f_line_world_list),
            flex_tau_line_world_list=tuple(flex_tau_line_world_list),
            gravity_rigid_list=tuple(gravity_rigid_list),
            magnet_params_list=tuple(magnet_params_list),
            coil_currents=coil_currents,
            magnetic_model=aux["magnetic_model"],
            scales=aux["scales"],
            cs_len=int(aux["cs_len"]),
            flex_block_offsets=tuple(aux["flex_block_offsets"]),
            flex_block_lens=tuple(aux["flex_block_lens"]),
            rigid_block_offsets=tuple(aux["rigid_block_offsets"]),
            rigid_block_lens=tuple(aux["rigid_block_lens"]),
            total_E_len=int(aux["total_E_len"]),
        )


def _C_S_flexible(x_n_bar: Array, x_np1_bar: Array) -> Array:
    Qn = x_n_bar[3:7]
    Qnp1 = x_np1_bar[3:7]
    return jnp.array([jnp.dot(Qn, Qn) - 1.0, jnp.dot(Qnp1, Qnp1) - 1.0], dtype=x_n_bar.dtype)


def _C_BV_proximal_pose(x_n_bar: Array, p0_bar: Array, Q0: Array) -> Array:
    return jnp.concatenate([x_n_bar[0:3] - p0_bar, x_n_bar[3:7] - Q0], axis=0)


def _C_BV_connect_prev_rigid(x_n_bar: Array, xR_prev_bar: Array) -> Array:
    return x_n_bar - xR_prev_bar


def _C_BV_distal_free_tip(x_np1_bar: Array) -> Array:
    return jnp.concatenate([x_np1_bar[7:10], x_np1_bar[10:13]], axis=0)


def residual_bar(z_bar: Array, params: SolverParams) -> Array:
    s = params.scales

    # -------- unpack z (same layout as your numpy nondim solver) --------
    x_nodes_list = []
    k_array_list = []
    x_rigid_list = []
    idx = 0

    N = len(params.flex)
    for i in range(N):
        M = params.M_list[i]
        nodes = []
        karr = []
        for _n in range(M + 1):
            nodes.append(z_bar[idx:idx + 13]); idx += 13
        for _n in range(M):
            karr.append(z_bar[idx:idx + 39]); idx += 39
        x_nodes_list.append(jnp.stack(nodes, axis=0))
        k_array_list.append(jnp.stack(karr, axis=0))
        x_rigid_list.append(z_bar[idx:idx + 13]); idx += 13

    # -------- build residual (same stacking order as numpy) --------
    E = jnp.zeros((params.total_E_len,), dtype=z_bar.dtype)

    # flex blocks
    offset = 0
    for i in range(N):
        flexp = params.flex[i]
        sbar_nodes = params.sbar_nodes[i]
        hbar_list = params.hbar_list[i]
        x_nodes = x_nodes_list[i]
        k_arr = k_array_list[i]
        M = params.M_list[i]
        L_seg = float(flexp.length)

        def rhs_bar(x_stage_bar: Array, sbar_stage: float) -> Array:
            sigma = sbar_stage * L_seg
            x_dim = x_bar_to_dim(x_stage_bar, s)

            f_line_world = params.flex_f_line_world_list[i]
            tau_line_world = params.flex_tau_line_world_list[i]

            dx_dsigma = cosserat_rhs_dim(
                x_dim, sigma,
                Kse_inv=flexp.Kse_inv,
                Kbt_inv=flexp.Kbt_inv,
                v_star=flexp.v_star,
                u_star=flexp.u_star,
                fext_density=(lambda _x, _s, v=f_line_world: v),
                tauext_density=(lambda _x, _s, v=tau_line_world: v),
            )
            return rhs_dim_to_rhs_bar_dsbar(dx_dsigma, L_seg=L_seg, s=s)

        for n in range(M):
            x_n = x_nodes[n]
            x_np1 = x_nodes[n + 1]
            # k_n = k_arr[n]
            k_n = k_arr[n].reshape((3, 13))  # <-- 关键修复：恢复 stage×state
            sbar_n = sbar_nodes[n]
            hbar = hbar_list[n]

            if (i == 0) and (n == 0):
                C_BV_fun = lambda xn, xnp1: _C_BV_proximal_pose(xn, params.p0_bar, params.Q0)
            elif (i > 0) and (n == 0):
                xR_prev = x_rigid_list[i - 1]
                C_BV_fun = lambda xn, xnp1, xR_prev=xR_prev: _C_BV_connect_prev_rigid(xn, xR_prev)
            else:
                C_BV_fun = lambda xn, xnp1: jnp.zeros((0,), dtype=z_bar.dtype)

            E_block = interval_residual_gl6_bar(
                x_n_bar=x_n,
                k_n_bar=k_n,
                x_np1_bar=x_np1,
                sbar_n=jnp.squeeze(jnp.asarray(sbar_n, dtype=z_bar.dtype)),
                hbar=jnp.squeeze(jnp.asarray(hbar, dtype=z_bar.dtype)),
                cs_len=params.cs_len,
                C_S_fun=_C_S_flexible,
                C_BV_fun=C_BV_fun,
                rhs_bar_fun=rhs_bar,
            )

            ln = E_block.shape[0]
            E = E.at[offset:offset + ln].set(E_block)
            offset += ln

    # rigid blocks (lumped totals)
    for i in range(N):
        rigidp = params.rigid[i]
        xR_bar = x_rigid_list[i]
        x_flex_end_bar = x_nodes_list[i][-1]

        # rigid proximal state in SI
        x_prox_dim = x_bar_to_dim(x_flex_end_bar, s)

        f_total_dim, tau_total_dim = compute_external_wrench_total_rigid(
            x_proximal=x_prox_dim,
            rigid_length=float(rigidp.length),
            gravity=params.gravity_rigid_list[i],
            magnetic_model=params.magnetic_model,
            magnet_params=params.magnet_params_list[i],
            coil_currents=params.coil_currents,
        )

        # apply additional user-provided totals if you kept them (optional)
        f_total_dim = f_total_dim + params.f_ext_list[i]
        tau_total_dim = tau_total_dim + params.tau_ext_list[i]

        # predicted rigid distal via analytic model
        xR_dim = x_bar_to_dim(xR_bar, s)
        xR_pred_dim = rigid_state_along_dim(
            x_prox_dim=x_prox_dim,
            sigma=jnp.asarray(float(rigidp.length), dtype=jnp.float64),
            rigid=rigidp,
            f_ext_total_dim=f_total_dim,
            tau_ext_total_dim=tau_total_dim,
        )

        # residual: xR_var - xR_pred, plus |Q| constraint and optional free tip
        C_S = jnp.array([jnp.dot(xR_bar[3:7], xR_bar[3:7]) - 1.0], dtype=z_bar.dtype)
        C_BV = jnp.zeros((0,), dtype=z_bar.dtype)
        if i == (N - 1):
            C_BV = _C_BV_distal_free_tip(xR_bar)

        # pack rigid block: [state(13), C_S(1), C_BV]
        xR_pred_bar = jnp.concatenate([
            xR_pred_dim[0:3] / s.L_ref,
            xR_pred_dim[3:7],
            xR_pred_dim[7:10] / s.F_ref,
            xR_pred_dim[10:13] / s.M_ref,
        ], axis=0)

        block = jnp.concatenate([xR_bar - xR_pred_bar, C_S, C_BV], axis=0)
        ln = block.shape[0]
        E = E.at[offset:offset + ln].set(block)
        offset += ln

    return E


# -------------------- Debug / Audit helpers (non-jit) --------------------
import numpy as _np
import jax
import jax.numpy as jnp

def _unpack_z_bar_for_debug(z_bar: jnp.ndarray, params) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray]]:
    """
    MUST match residual_bar() layout:
      For each segment i:
        nodes: (M+1)*13
        k:     M*39
        rigid: 13
    """
    z_bar = jnp.asarray(z_bar, dtype=jnp.float64)
    x_nodes_list = []
    k_array_list = []
    x_rigid_list = []
    idx = 0
    N = len(params.flex)

    for i in range(N):
        M = int(params.M_list[i])
        nodes = []
        karr = []
        for _ in range(M + 1):
            nodes.append(z_bar[idx:idx + 13]); idx += 13
        for _ in range(M):
            karr.append(z_bar[idx:idx + 39]); idx += 39
        x_nodes_list.append(jnp.stack(nodes, axis=0))
        k_array_list.append(jnp.stack(karr, axis=0))
        x_rigid_list.append(z_bar[idx:idx + 13]); idx += 13

    return x_nodes_list, k_array_list, x_rigid_list


def audit_last_rigid_force_identity(
    z_bar: jnp.ndarray,
    params,
    *,
    it: int | None = None,
    seg_idx: int | None = None,
    print_vectors: bool = True,
) -> None:
    """
    Audit for rigid segment force/moment balance identity at distal prediction.

    Checks (in BAR units):
      inferred_pred = f_prox_bar - f_end_pred_bar  ≈ f_total_bar
      inferred_var  = f_prox_bar - f_end_var_bar   ≈ f_total_bar

    Same for tau.
    """
    s = params.scales
    N = len(params.rigid)
    if seg_idx is None:
        seg_idx = N - 1

    x_nodes_list, _, x_rigid_list = _unpack_z_bar_for_debug(z_bar, params)

    rigidp = params.rigid[seg_idx]
    xR_var_bar = x_rigid_list[seg_idx]
    x_flex_end_bar = x_nodes_list[seg_idx][-1]  # proximal of rigid

    # proximal state in SI (for load computation + rigid analytic)
    x_prox_dim = x_bar_to_dim(x_flex_end_bar, s)

    # totals in SI (world, about proximal)
    f_total_dim, tau_total_dim = compute_external_wrench_total_rigid(
        x_proximal=x_prox_dim,
        rigid_length=float(rigidp.length),
        gravity=params.gravity_rigid_list[seg_idx],
        magnetic_model=params.magnetic_model,
        magnet_params=params.magnet_params_list[seg_idx],
        coil_currents=params.coil_currents,
    )
    # add optional user totals (same as residual_bar)
    f_total_dim = f_total_dim + params.f_ext_list[seg_idx]
    tau_total_dim = tau_total_dim + params.tau_ext_list[seg_idx]

    # rigid distal prediction in SI
    xR_pred_dim = rigid_state_along_dim(
        x_prox_dim=x_prox_dim,
        sigma=jnp.asarray(float(rigidp.length), dtype=jnp.float64),
        # v_star=rigidp.v_star,
        rigid=rigidp,
        f_ext_total_dim=f_total_dim,
        tau_ext_total_dim=tau_total_dim,
    )

    # pack pred to BAR (same as residual_bar)
    xR_pred_bar = jnp.concatenate([
        xR_pred_dim[0:3] / s.L_ref,
        xR_pred_dim[3:7],
        xR_pred_dim[7:10] / s.F_ref,
        xR_pred_dim[10:13] / s.M_ref,
    ], axis=0)

    # -------------------- audit numbers (BAR) --------------------
    f_prox_bar = x_flex_end_bar[7:10]
    tau_prox_bar = x_flex_end_bar[10:13]

    f_end_pred_bar = xR_pred_bar[7:10]
    tau_end_pred_bar = xR_pred_bar[10:13]

    f_end_var_bar = xR_var_bar[7:10]
    tau_end_var_bar = xR_var_bar[10:13]

    f_total_bar = f_total_dim / s.F_ref
    tau_total_bar = tau_total_dim / s.M_ref

    inferred_pred = f_prox_bar - f_end_pred_bar
    inferred_var  = f_prox_bar - f_end_var_bar

    inferred_tau_pred = tau_prox_bar - tau_end_pred_bar
    inferred_tau_var  = tau_prox_bar - tau_end_var_bar

    diff_pred = inferred_pred - f_total_bar
    diff_var  = inferred_var  - f_total_bar

    diff_tau_pred = inferred_tau_pred - tau_total_bar
    diff_tau_var  = inferred_tau_var  - tau_total_bar

    # block residual magnitude (state part only)
    block_state = xR_var_bar - xR_pred_bar

    # device->host for printing
    def _hg(x):
        return _np.asarray(jax.device_get(x), dtype=_np.float64)

    tag = f"[it={it}] " if it is not None else ""
    print(f"{tag}[RIGID AUDIT LAST] seg={seg_idx}  L={float(rigidp.length):.6g} m")
    print(f"{tag}  ||xR_var - xR_pred||_bar = {float(jnp.linalg.norm(block_state)):.6g}")
    print(f"{tag}  f_total_dim (SI)= {_hg(f_total_dim)}")
    print(f"{tag}  tau_total_dim(SI)= {_hg(tau_total_dim)}")
    print(f"{tag}  ||f_total_bar||={float(jnp.linalg.norm(f_total_bar)):.6g}   ||tau_total_bar||={float(jnp.linalg.norm(tau_total_bar)):.6g}")

    print(f"{tag}  inferred_pred_bar = f_prox - f_end_pred : ||·||={float(jnp.linalg.norm(inferred_pred)):.6g}")
    print(f"{tag}  diff_pred_bar     = inferred_pred - f_total_bar : ||·||={float(jnp.linalg.norm(diff_pred)):.6g}")

    print(f"{tag}  inferred_var_bar  = f_prox - f_end_var  : ||·||={float(jnp.linalg.norm(inferred_var)):.6g}")
    print(f"{tag}  diff_var_bar      = inferred_var - f_total_bar  : ||·||={float(jnp.linalg.norm(diff_var)):.6g}")

    print(f"{tag}  inferred_tau_pred_bar = tau_prox - tau_end_pred : ||·||={float(jnp.linalg.norm(inferred_tau_pred)):.6g}")
    print(f"{tag}  diff_tau_pred_bar     = inferred_tau_pred - tau_total_bar : ||·||={float(jnp.linalg.norm(diff_tau_pred)):.6g}")

    print(f"{tag}  inferred_tau_var_bar  = tau_prox - tau_end_var  : ||·||={float(jnp.linalg.norm(inferred_tau_var)):.6g}")
    print(f"{tag}  diff_tau_var_bar      = inferred_tau_var - tau_total_bar  : ||·||={float(jnp.linalg.norm(diff_tau_var)):.6g}")

    # free-tip BV sanity (last rigid has C_BV = [f_end, tau_end] in BAR)
    print(f"{tag}  free-tip check (var): ||f_end_var_bar||={float(jnp.linalg.norm(f_end_var_bar)):.6g}  ||tau_end_var_bar||={float(jnp.linalg.norm(tau_end_var_bar)):.6g}")

    if print_vectors:
        print(f"{tag}  f_prox_bar     = {_hg(f_prox_bar)}")
        print(f"{tag}  f_end_pred_bar = {_hg(f_end_pred_bar)}")
        print(f"{tag}  f_end_var_bar  = {_hg(f_end_var_bar)}")
        print(f"{tag}  f_total_bar    = {_hg(f_total_bar)}")
        print(f"{tag}  diff_pred_bar  = {_hg(diff_pred)}")
        print(f"{tag}  diff_var_bar   = {_hg(diff_var)}")

        print(f"{tag}  tau_prox_bar     = {_hg(tau_prox_bar)}")
        print(f"{tag}  tau_end_pred_bar = {_hg(tau_end_pred_bar)}")
        print(f"{tag}  tau_end_var_bar  = {_hg(tau_end_var_bar)}")
        print(f"{tag}  tau_total_bar    = {_hg(tau_total_bar)}")
        print(f"{tag}  diff_tau_pred_bar= {_hg(diff_tau_pred)}")
        print(f"{tag}  diff_tau_var_bar = {_hg(diff_tau_var)}")



# ----------------------------- JAX block diagnostics -----------------------------
def _scalar_norm(x: Array) -> float:
    return float(jnp.linalg.norm(x))

def _scalar_maxabs(x: Array) -> float:
    return float(jnp.max(jnp.abs(x))) if x.size > 0 else 0.0

def _fmt_small_list(vals, ndigits: int = 6) -> str:
    return "[" + ", ".join([f"{float(v):.{ndigits}g}" for v in vals]) + "]"

def print_top_blocks_jax(
    E: Array,
    params: SolverParams,
    *,
    topk_flex: int = 9,
    topk_rigid: int = 3,
    verbose: bool = True,
    title_prefix: str = "",
) -> None:
    """
    Diagnose which residual blocks dominate ||E|| for the JAX nondim solver.

    Flex block layout (per interval):
      [ C_S (cs_len), res_state (13), res_ks (39 = 3*13), C_BV (0/7/13) ]
    Rigid block layout (per segment):
      [ core(13), quat_norm(1), BV(0/6) ]
    """
    cs_len = int(params.cs_len)
    N = len(params.flex)

    total = _scalar_norm(E)
    print(f"{title_prefix}total {total}")

    # -------- flex blocks --------
    flex_stats = []
    flex_block_idx = 0
    for seg in range(N):
        M = int(params.M_list[seg])
        for interval in range(M):
            off = int(params.flex_block_offsets[flex_block_idx])
            ln = int(params.flex_block_lens[flex_block_idx])
            blk = E[off:off+ln]

            flex_stats.append(dict(
                seg=seg,
                interval=interval,
                idx=flex_block_idx,
                off=off,
                ln=ln,
                norm=_scalar_norm(blk),
                maxabs=_scalar_maxabs(blk),
            ))
            flex_block_idx += 1

    flex_stats.sort(key=lambda d: d["norm"], reverse=True)

    if verbose:
        print(f"{title_prefix}Top flex blocks (verbose):")
    else:
        print(f"{title_prefix}Top flex blocks:")

    for k, d in enumerate(flex_stats[:topk_flex]):
        off, ln = d["off"], d["ln"]
        blk = E[off:off+ln]

        C_S = blk[:cs_len]
        state = blk[cs_len:cs_len+13]
        ks = blk[cs_len+13:cs_len+13+39]
        BV = blk[cs_len+13+39:]

        # state breakdown
        p = state[0:3]
        Q = state[3:7]
        f = state[7:10]
        tau = state[10:13]

        # ks breakdown (3 stages x 13)
        ks0 = ks[0:13]
        ks1 = ks[13:26]
        ks2 = ks[26:39]
        stages = [_scalar_norm(ks0), _scalar_norm(ks1), _scalar_norm(ks2)]

        if verbose:
            print(
                f"{title_prefix}  seg={d['seg']} interval={d['interval']} len={ln} "
                f"||E||={d['norm']:.6g} max={d['maxabs']:.6g} | "
                f"C_S={_scalar_norm(C_S):.3g}, state={_scalar_norm(state):.4g}, "
                f"ks={_scalar_norm(ks):.4g} (stages {_fmt_small_list(stages, ndigits=4)}), "
                f"BV={_scalar_norm(BV):.4g} (bv_len={int(BV.size)})"
            )
        else:
            print(
                f"{title_prefix}  seg={d['seg']} interval={d['interval']} "
                f"||E||={d['norm']:.6g} max={d['maxabs']:.6g}"
            )

    # -------- rigid blocks --------
    rigid_stats = []
    for seg in range(N):
        off = int(params.rigid_block_offsets[seg])
        ln = int(params.rigid_block_lens[seg])
        blk = E[off:off+ln]
        rigid_stats.append(dict(
            seg=seg,
            off=off,
            ln=ln,
            norm=_scalar_norm(blk),
            maxabs=_scalar_maxabs(blk),
        ))
    rigid_stats.sort(key=lambda d: d["norm"], reverse=True)

    print(f"{title_prefix}Rigid blocks:")
    for d in rigid_stats[:topk_rigid]:
        off, ln = d["off"], d["ln"]
        blk = E[off:off+ln]
        core = blk[:13]
        quat = blk[13:14]
        BV = blk[14:]

        if verbose:
            # core breakdown
            p = core[0:3]
            Q = core[3:7]
            f = core[7:10]
            tau = core[10:13]
            print(
                f"{title_prefix}  seg={d['seg']} len={ln} ||E||={d['norm']:.6g} max={d['maxabs']:.6g} | "
                f"core={_scalar_norm(core):.4g} (p={_scalar_norm(p):.3g}, Q={_scalar_norm(Q):.3g}, "
                f"f={_scalar_norm(f):.3g}, tau={_scalar_norm(tau):.3g}), "
                f"quat={_scalar_norm(quat):.3g}, BV={_scalar_norm(BV):.3g} (bv_len={int(BV.size)})"
            )
        else:
            print(
                f"{title_prefix}  seg={d['seg']} len={ln} ||E||={d['norm']:.6g} max={d['maxabs']:.6g}"
            )




@dataclass
class LMStats:
    """Diagnostics and linearization artifacts returned by :meth:`solve_lm`.

    This is intentionally lightweight and aimed at IK / debugging.
    """

    z_star: Array
    ok_strict: bool
    E_star: Array
    normE: float
    cost: float
    lam: float
    J: Optional[Array]
    jac_method: str
    stop_reason: str
    n_iter: int


class MultiSegmentEquilibriumSolverNondimJAX:
    def __init__(self, params: SolverParams):
        self.params = params
        self.residual_jit = jax.jit(lambda z: residual_bar(z, self.params))

    def jacobian_jit(self, z: Array, *, method: str = "fwd") -> Array:
        if method == "fwd":
            return jax.jacfwd(self.residual_jit)(z)
        if method == "rev":
            return jax.jacrev(self.residual_jit)(z)
        raise ValueError("jac_method must be 'fwd' or 'rev'")

    def _pack_return(self, z: Array, ok: bool, *, tol: float, lam: float, jac_method: str, stop_reason: str,
                     n_iter: int, return_stats: bool) -> tuple:
        if not return_stats:
            return (jnp.array(z), bool(ok))
        E = self.residual_jit(z)
        normE = float(jnp.linalg.norm(E))
        cost = 0.5 * normE * normE
        # Always recompute a Jacobian at the returned z to keep it consistent.
        J = self.jacobian_jit(z, method=jac_method)
        stats = LMStats(
            z_star=jnp.array(z),
            ok_strict=bool(ok and (normE < tol)),
            E_star=jnp.array(E),
            normE=float(normE),
            cost=float(cost),
            lam=float(lam),
            J=jnp.array(J),
            jac_method=str(jac_method),
            stop_reason=str(stop_reason),
            n_iter=int(n_iter),
        )
        return (jnp.array(z), bool(ok), stats)

    def solve_lm(
        self,
        z0_bar: Array,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        lm_damping: float = 1e-3,
        jac_method: str = "fwd",
        callback: Optional[Callable[[int, Array, float], None]] = None,
        # --- smart stopping ---
        rel_cost_tol: float = 1e-12,
        abs_cost_tol: float = 1e-18,
        patience: int = 20,
        reject_patience: int = 10,
        gtol: float = 1e-12,
        xtol: float = 1e-12,
        lam_max: float = 1e10,
        return_stats: bool = False,
    ):
        """Levenberg–Marquardt solver with smarter termination.

        Return values
        -------------
        - If ``return_stats=False`` (default): returns ``(z_star_bar, ok_strict)``.
        - If ``return_stats=True``: returns ``(z_star_bar, ok_strict, stats)``.
        """
        z = jnp.asarray(z0_bar, dtype=jnp.float64)
        lam = float(lm_damping)

        # Initial residual
        E = self.residual_jit(z)
        normE = float(jnp.linalg.norm(E))
        cost = 0.5 * normE * normE

        # Book-keeping for smart termination
        best_cost = float(cost)
        no_improve = 0  # counts accepted steps with insufficient improvement
        consecutive_reject = 0  # counts outer iterations with all 5 attempts rejected

        print(f"[Multi-LM-JAX] iter=0, ||E||={normE:.3e}, lambda={lam:.3e}")
        if callback is not None:
            callback(0, jnp.array(z), normE)

        # NaN/Inf guard
        if (not jnp.isfinite(normE)) or (not jnp.isfinite(cost)):
            print("[Multi-LM-JAX] numerical error at init (NaN/Inf).")
            return self._pack_return(z, False, tol=tol, lam=lam, jac_method=jac_method,
                                     stop_reason="nan_init", n_iter=0, return_stats=return_stats)

        for it in range(max_iter):
            # (A) classic residual stop
            if normE < tol:
                print(f"[Multi-LM-JAX] converged. Final ||E||={normE:.3e}")
                return self._pack_return(z, True, tol=tol, lam=lam, jac_method=jac_method,
                                         stop_reason="converged", n_iter=it+1, return_stats=return_stats)

            # Jacobian + normal equations pieces
            J = self.jacobian_jit(z, method=jac_method)
            Jt = jnp.transpose(J)
            JtJ = Jt @ J
            g = Jt @ E

            # (B) first-order optimality stop: ||J^T E|| small
            g_norm_inf = float(jnp.max(jnp.abs(g))) if g.size > 0 else 0.0
            if g_norm_inf < gtol:
                print(
                    f"[Multi-LM-JAX] stop: small gradient ||J^T E||_inf={g_norm_inf:.3e} < gtol={gtol:.3e}"
                )
                return self._pack_return(z, False, tol=tol, lam=lam, jac_method=jac_method,
                                         stop_reason="small_grad", n_iter=it, return_stats=return_stats)

            I = jnp.eye(JtJ.shape[0], dtype=JtJ.dtype)

            step_accepted = False
            last_delta_norm = None

            # Up to 5 damping attempts per outer iteration
            for _ in range(5):
                A = JtJ + lam * I
                delta = jnp.linalg.solve(A, g)
                last_delta_norm = float(jnp.linalg.norm(delta))

                # (C) step-size stop: delta too small
                z_norm = float(jnp.linalg.norm(z))
                if last_delta_norm < xtol * (xtol + z_norm):
                    print(f"[Multi-LM-JAX] stop: tiny step ||delta||={last_delta_norm:.3e}")
                    return self._pack_return(z, False, tol=tol, lam=lam, jac_method=jac_method,
                                             stop_reason="tiny_step", n_iter=it, return_stats=return_stats)

                # predicted reduction
                pred = float((g @ delta) - 0.5 * (delta @ (JtJ @ delta)))

                z_new = z - delta
                E_new = self.residual_jit(z_new)
                normE_new = float(jnp.linalg.norm(E_new))
                cost_new = 0.5 * normE_new * normE_new

                # NaN/Inf guard: treat as rejected and increase damping
                if (not jnp.isfinite(normE_new)) or (not jnp.isfinite(cost_new)):
                    lam = min(lam * 3.0, lam_max)
                    continue

                actual = cost - cost_new
                rho = actual / pred if abs(pred) > 1e-15 else 0.0

                if rho > 0.0:
                    # Accept
                    z, E, normE, cost = z_new, E_new, normE_new, cost_new
                    step_accepted = True
                    consecutive_reject = 0

                    # LM damping update (keep your original policy)
                    if rho > 0.75:
                        lam *= 0.3
                        status = "↓↓"
                    elif rho > 0.25:
                        lam *= 0.5
                        status = "↓"
                    else:
                        status = "→"
                    lam = max(min(lam, lam_max), 1e-12)

                    # (D) stagnation tracking: improvement measured on accepted steps
                    rel_impr = (best_cost - cost) / max(best_cost, 1e-30)
                    abs_impr = (best_cost - cost)
                    if (rel_impr > rel_cost_tol) or (abs_impr > abs_cost_tol):
                        best_cost = cost
                        no_improve = 0
                    else:
                        no_improve += 1

                    print(
                        f"[Multi-LM-JAX] iter={it+1}, ||E||={normE:.3e}, lambda={lam:.3e} {status}, "
                        f"rho={rho:.2f}, no_improve={no_improve}/{patience}"
                    )
                    if callback is not None:
                        callback(it + 1, jnp.array(z), normE)

                    # (E) stagnation stop
                    if no_improve >= patience:
                        print(
                            f"[Multi-LM-JAX] stop: stagnation (no sufficient improvement for {patience} accepted steps)."
                        )
                        return self._pack_return(z, False, tol=tol, lam=lam, jac_method=jac_method,
                                                 stop_reason="stagnation", n_iter=it+1, return_stats=return_stats)

                    break

                # Reject: increase damping
                lam = min(lam * 3.0, lam_max)

            if not step_accepted:
                consecutive_reject += 1
                print(
                    f"[Multi-LM-JAX] iter={it+1}, ||E||={normE:.3e}, lambda={lam:.3e} x (Rejected) "
                    f"reject={consecutive_reject}/{reject_patience}"
                )

                # (F) reject-based failure stop
                if (consecutive_reject >= reject_patience) or (lam >= lam_max * 0.999):
                    print("[Multi-LM-JAX] stop: too many rejected iterations or lambda saturated.")
                    return self._pack_return(z, False, tol=tol, lam=lam, jac_method=jac_method,
                                             stop_reason="reject_or_lammax", n_iter=it+1, return_stats=return_stats)

        print(f"[Multi-LM-JAX] Reached max iterations. Final ||E||={normE:.3e}")
        return self._pack_return(z, False, tol=tol, lam=lam, jac_method=jac_method,
                              stop_reason="max_iter", n_iter=max_iter, return_stats=return_stats)


class MultiSegmentEquilibriumSolverNondimJAXCached:
    """A reusable LM solver that avoids rebuilding JAX programs per FK call.

    Why this exists
    --------------
    The original :class:`MultiSegmentEquilibriumSolverNondimJAX` stores ``params`` in
    ``self`` and builds jitted closures like ``jit(lambda z: residual_bar(z, self.params))``.
    If you create a new solver instance (and thus a new closure) every IK iteration,
    JAX will typically retrace/recompile, which is expensive.

    This cached variant **does not capture params in a closure**. Instead, all
    compiled functions take ``(z, params)`` as explicit arguments, so the compiled
    executables can be reused across repeated calls as long as shapes/dtypes are
    unchanged.

    Notes
    -----
    - ``SolverParams`` is a PyTree, so passing it as an argument is JAX-friendly.
    - If you change discretization sizes (M_list), the shapes change and JAX must
      compile a new executable. For IK, M_list is fixed.
    - ``NondimScales`` is kept as a *static* field in ``SolverParams`` aux-data.
      If you change ``L_protrude`` (thus scales), JAX may retrace. For the current
      IK scripts, ``L_protrude`` is fixed, so this is a one-time compile.
    """

    def __init__(self):
        # Residual and Jacobians (compiled once per shape/static-config)
        self.residual_jit = jax.jit(residual_bar)
        self._jac_fwd = jax.jit(jax.jacfwd(residual_bar, argnums=0))
        self._jac_rev = jax.jit(jax.jacrev(residual_bar, argnums=0))

    def jacobian_jit(self, z: Array, params: SolverParams, *, method: str = "fwd") -> Array:
        if method == "fwd":
            return self._jac_fwd(z, params)
        if method == "rev":
            return self._jac_rev(z, params)
        raise ValueError("jac_method must be 'fwd' or 'rev'")

    def _pack_return(self, z: Array, params: SolverParams, ok: bool, *, tol: float, lam: float,
                     jac_method: str, stop_reason: str, n_iter: int, return_stats: bool) -> tuple:
        if not return_stats:
            return (jnp.array(z), bool(ok))
        E = self.residual_jit(z, params)
        normE = float(jnp.linalg.norm(E))
        cost = 0.5 * normE * normE
        J = self.jacobian_jit(z, params, method=jac_method)
        stats = LMStats(
            z_star=jnp.array(z),
            ok_strict=bool(ok and (normE < tol)),
            E_star=jnp.array(E),
            normE=float(normE),
            cost=float(cost),
            lam=float(lam),
            J=jnp.array(J),
            jac_method=str(jac_method),
            stop_reason=str(stop_reason),
            n_iter=int(n_iter),
        )
        return (jnp.array(z), bool(ok), stats)

    def solve_lm(
        self,
        z0_bar: Array,
        params: SolverParams,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        lm_damping: float = 1e-3,
        jac_method: str = "fwd",
        callback: Optional[Callable[[int, Array, float], None]] = None,
        # --- smart stopping ---
        rel_cost_tol: float = 1e-12,
        abs_cost_tol: float = 1e-18,
        patience: int = 20,
        reject_patience: int = 10,
        gtol: float = 1e-12,
        xtol: float = 1e-12,
        lam_max: float = 1e10,
        return_stats: bool = False,
        verbose: bool = True,
    ):
        """Same algorithm as :meth:`MultiSegmentEquilibriumSolverNondimJAX.solve_lm`.

        The only difference is that ``params`` is an explicit argument, enabling
        compilation reuse across FK calls.
        """
        z = jnp.asarray(z0_bar, dtype=jnp.float64)
        lam = float(lm_damping)

        E = self.residual_jit(z, params)
        normE = float(jnp.linalg.norm(E))
        cost = 0.5 * normE * normE

        best_cost = float(cost)
        no_improve = 0
        consecutive_reject = 0

        if verbose:
            print(f"[Multi-LM-JAX] iter=0, ||E||={normE:.3e}, lambda={lam:.3e}")
        if callback is not None:
            callback(0, jnp.array(z), normE)

        if (not jnp.isfinite(normE)) or (not jnp.isfinite(cost)):
            if verbose:
                print("[Multi-LM-JAX] numerical error at init (NaN/Inf).")
            return self._pack_return(z, params, False, tol=tol, lam=lam, jac_method=jac_method,
                                     stop_reason="nan_init", n_iter=0, return_stats=return_stats)

        for it in range(max_iter):
            if normE < tol:
                if verbose:
                    print(f"[Multi-LM-JAX] converged. Final ||E||={normE:.3e}")
                return self._pack_return(z, params, True, tol=tol, lam=lam, jac_method=jac_method,
                                         stop_reason="converged", n_iter=it+1, return_stats=return_stats)

            J = self.jacobian_jit(z, params, method=jac_method)
            g = jnp.transpose(J) @ E
            g_norm = float(jnp.linalg.norm(g))

            if g_norm < gtol:
                if verbose:
                    print(f"[Multi-LM-JAX] stop: small gradient ||J^T E||={g_norm:.3e} < {gtol:.3e}")
                return self._pack_return(z, params, True, tol=tol, lam=lam, jac_method=jac_method,
                                         stop_reason="gtol", n_iter=it+1, return_stats=return_stats)

            A = jnp.transpose(J) @ J + lam * jnp.eye(J.shape[1], dtype=J.dtype)
            dz = jnp.linalg.solve(A, -g)
            dz_norm = float(jnp.linalg.norm(dz))
            if dz_norm < xtol:
                if verbose:
                    print(f"[Multi-LM-JAX] stop: small step ||dz||={dz_norm:.3e} < {xtol:.3e}")
                return self._pack_return(z, params, True, tol=tol, lam=lam, jac_method=jac_method,
                                         stop_reason="xtol", n_iter=it+1, return_stats=return_stats)

            z_new = z + dz
            E_new = self.residual_jit(z_new, params)
            normE_new = float(jnp.linalg.norm(E_new))
            cost_new = 0.5 * normE_new * normE_new

            if (not jnp.isfinite(cost_new)) or (not jnp.isfinite(normE_new)):
                lam = min(lam * 10.0, lam_max)
                consecutive_reject += 1
                if verbose:
                    print(f"[Multi-LM-JAX] iter={it+1}, NaN/Inf -> lambda={lam:.3e} reject")
                if consecutive_reject >= reject_patience:
                    return self._pack_return(z, params, False, tol=tol, lam=lam, jac_method=jac_method,
                                             stop_reason="nan_reject_patience", n_iter=it+1, return_stats=return_stats)
                continue

            # Predicted reduction (standard LM model)
            pred_red = 0.5 * float(dz @ (lam * dz - g))
            act_red = float(cost - cost_new)
            rho = act_red / (pred_red + 1e-30)

            if act_red > 0:
                # accept
                z, E, normE, cost = z_new, E_new, normE_new, cost_new
                consecutive_reject = 0
                if cost < best_cost - max(abs_cost_tol, rel_cost_tol * best_cost):
                    best_cost = float(cost)
                    no_improve = 0
                else:
                    no_improve += 1

                if rho > 0.75:
                    lam *= 0.3
                elif rho < 0.25:
                    lam *= 10.0
                lam = float(min(max(lam, 1e-30), lam_max))

                if verbose and (it + 1) % 1000 == 0:
                    # keep the same format as original class (approx.)
                    arrow = "↓↓" if rho > 0.75 else ("↑↑" if rho < 0.25 else "--")
                    print(f"[Multi-LM-JAX] iter={it+1}, ||E||={normE:.3e}, lambda={lam:.3e} {arrow}, rho={rho:.2f}, no_improve={no_improve}/{patience}")

                if no_improve >= patience:
                    if verbose:
                        print("[Multi-LM-JAX] stop: no-improve patience reached")
                    return self._pack_return(z, params, True, tol=tol, lam=lam, jac_method=jac_method,
                                             stop_reason="patience", n_iter=it+1, return_stats=return_stats)
            else:
                # reject
                consecutive_reject += 1
                lam = min(lam * 10.0, lam_max)
                if verbose:
                    print(f"[Multi-LM-JAX] iter={it+1}, reject, ||E||={normE:.3e}, lambda={lam:.3e}, rho={rho:.2f}")

                if consecutive_reject >= reject_patience:
                    if verbose:
                        print("[Multi-LM-JAX] stop: reject-patience reached")
                    return self._pack_return(z, params, False, tol=tol, lam=lam, jac_method=jac_method,
                                             stop_reason="reject_patience", n_iter=it+1, return_stats=return_stats)

        if verbose:
            print("[Multi-LM-JAX] stop: max_iter")
        return self._pack_return(z, params, False, tol=tol, lam=lam, jac_method=jac_method,
                                 stop_reason="max_iter", n_iter=max_iter, return_stats=return_stats)


# -------------------- Convenience builder --------------------

def build_solver_params_from_numpy_solver(
    np_solver,
    *,
    magnetic_model: Optional[MagneticModel] = None,
    coil_currents: Optional[Array] = None,
    flex_f_line_world_list: Optional[Tuple[Array, ...]] = None,
    flex_tau_line_world_list: Optional[Tuple[Array, ...]] = None,
) -> SolverParams:
    """Build JAX SolverParams from your existing numpy nondim solver instance."""

    flex_params = []
    rigid_params = []
    sbar_nodes = []
    hbar_list = []
    M_list = []

    for flex, mesh in zip(np_solver.flex_segs, np_solver.meshes):
        Kse_inv = jnp.linalg.inv(jnp.asarray(flex.K_se, dtype=jnp.float64))
        Kbt_inv = jnp.linalg.inv(jnp.asarray(flex.K_bt, dtype=jnp.float64))
        flex_params.append(FlexibleParams(
            length=float(flex.length),
            Kse_inv=Kse_inv,
            Kbt_inv=Kbt_inv,
            v_star=jnp.asarray(flex.v_star, dtype=jnp.float64).reshape(3,),
            u_star=jnp.asarray(flex.u_star, dtype=jnp.float64).reshape(3,),
        ))
        sbar_nodes.append(jnp.asarray(mesh.sbar_nodes, dtype=jnp.float64))
        hbar_list.append(jnp.asarray(mesh.hbar_list, dtype=jnp.float64))
        M_list.append(int(mesh.M))

    for rigid in np_solver.rigid_segs:
        rigid_params.append(RigidParams(
            length=float(rigid.length),
            v_star=jnp.asarray(rigid.v_star, dtype=jnp.float64).reshape(3,),
        ))

    scales = NondimScales(
        L_ref=float(np_solver.scales.L_ref),
        F_ref=float(np_solver.scales.F_ref),
        M_ref=float(np_solver.scales.M_ref),
    )

    p0_bar = jnp.asarray(np_solver.p0_target, dtype=jnp.float64).reshape(3,) / scales.L_ref
    Q0 = jnp.asarray(np_solver.Q0_target, dtype=jnp.float64).reshape(4,)

    f_ext_list = tuple(jnp.asarray(f, dtype=jnp.float64).reshape(3,) for f in np_solver.f_ext_list)
    tau_ext_list = tuple(jnp.asarray(t, dtype=jnp.float64).reshape(3,) for t in np_solver.tau_ext_list)

    # flexible distributed loads (world frame)
    N = len(flex_params)
    if flex_f_line_world_list is None:
        flex_f_line_world_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))
    if flex_tau_line_world_list is None:
        flex_tau_line_world_list = tuple(jnp.zeros((3,), dtype=jnp.float64) for _ in range(N))

    flex_f_line_world_list = tuple(jnp.asarray(v, dtype=jnp.float64).reshape(3,) for v in flex_f_line_world_list)
    flex_tau_line_world_list = tuple(jnp.asarray(v, dtype=jnp.float64).reshape(3,) for v in flex_tau_line_world_list)

    # gravity list
    grav_list = []
    if getattr(np_solver, "gravity_rigid_list", None) is None:
        grav_list = [None for _ in range(len(rigid_params))]
    else:
        for g in np_solver.gravity_rigid_list:
            if g is None:
                grav_list.append(None)
            else:
                grav_list.append(GravityRigid(
                    mass=float(g.mass),
                    g_world=jnp.asarray(g.g_world, dtype=jnp.float64).reshape(3,),
                    r_cm_body=jnp.asarray(g.r_cm_body, dtype=jnp.float64).reshape(3,),
                ))

    magnet_params_list = []
    if getattr(np_solver, "magnet_params_list", None) is None:
        magnet_params_list = [None for _ in range(len(rigid_params))]
    else:
        magnet_params_list = list(np_solver.magnet_params_list)

    # residual layout (must match numpy assembly order)
    cs_len = 2
    flex_block_offsets = []
    flex_block_lens = []
    rigid_block_offsets = []
    rigid_block_lens = []
    offset = 0

    for i in range(N):
        M = M_list[i]
        for n in range(M):
            if (i == 0) and (n == 0):
                bv_len = 7
            elif (i > 0) and (n == 0):
                bv_len = 13
            else:
                bv_len = 0
            ln = cs_len + 13 + 39 + bv_len
            flex_block_offsets.append(offset)
            flex_block_lens.append(ln)
            offset += ln

    for i in range(N):
        bv_len = 6 if (i == N - 1) else 0
        ln = 13 + 1 + bv_len
        rigid_block_offsets.append(offset)
        rigid_block_lens.append(ln)
        offset += ln

    return SolverParams(
        flex=tuple(flex_params),
        rigid=tuple(rigid_params),
        sbar_nodes=tuple(sbar_nodes),
        hbar_list=tuple(hbar_list),
        M_list=tuple(M_list),
        p0_bar=p0_bar,
        Q0=Q0,
        f_ext_list=f_ext_list,
        tau_ext_list=tau_ext_list,
        flex_f_line_world_list=tuple(flex_f_line_world_list),
        flex_tau_line_world_list=tuple(flex_tau_line_world_list),
        gravity_rigid_list=tuple(grav_list),
        magnet_params_list=tuple(magnet_params_list),
        coil_currents=None if coil_currents is None else jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1,),
        magnetic_model=magnetic_model,
        scales=scales,
        cs_len=cs_len,
        flex_block_offsets=tuple(int(x) for x in flex_block_offsets),
        flex_block_lens=tuple(int(x) for x in flex_block_lens),
        rigid_block_offsets=tuple(int(x) for x in rigid_block_offsets),
        rigid_block_lens=tuple(int(x) for x in rigid_block_lens),
        total_E_len=int(offset),
    )
