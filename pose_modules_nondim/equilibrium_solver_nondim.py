# equilibrium_solver_nondim.py
import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass

from .segments_nondim import RigidSegment, FlexibleSegment, GL3_A, GL3_B, GL3_C
from .rod_mesh_nondim import RodMesh
from .utils_nondim import make_C_S_flexible, make_C_BV_proximal_pose, make_C_BV_distal_free_tip
from .external_wrench_nondim import GravityRigid, MagneticModel, compute_external_wrench_total_rigid

from .nondim import (
    NondimScales,
    compute_default_scales,
    x_bar_to_dim,
    x_dim_to_bar,
    rhs_dim_to_rhs_bar_dsbar,
)


@dataclass
class MultiSegmentEquilibriumSolverNondim:
    flex_segs: List[FlexibleSegment]
    meshes: List[RodMesh]
    rigid_segs: List[RigidSegment]

    p0_target: np.ndarray
    Q0_target: np.ndarray

    f_ext_list: Optional[List[np.ndarray]] = None
    tau_ext_list: Optional[List[np.ndarray]] = None

    gravity_rigid_list: Optional[List[GravityRigid]] = None

    magnetic_model: Optional[MagneticModel] = None
    magnet_params_list: Optional[List[dict]] = None
    coil_currents: Optional[np.ndarray] = None

    # nondim scales
    scales: Optional[NondimScales] = None

    max_iter: int = 100000
    tol: float = 1e-6
    lm_damping: float = 1e-3

    # debug controls
    debug_print: bool = True
    debug_audit_rigid_force: bool = True
    debug_audit_seg_idx: int = 0

    def __post_init__(self):
        self.N = len(self.flex_segs)
        assert self.N == len(self.meshes) == len(self.rigid_segs)

        if self.f_ext_list is None:
            self.f_ext_list = [np.zeros(3) for _ in range(self.N)]
        if self.tau_ext_list is None:
            self.tau_ext_list = [np.zeros(3) for _ in range(self.N)]

        assert len(self.f_ext_list) == self.N
        assert len(self.tau_ext_list) == self.N

        if self.gravity_rigid_list is not None:
            assert len(self.gravity_rigid_list) == self.N

        if self.magnet_params_list is not None:
            assert len(self.magnet_params_list) == self.N

        if self.scales is None:
            self.scales = compute_default_scales(self.flex_segs, self.rigid_segs)

        # proximal pose constraint in BAR space: p0_bar = p0 / L_ref
        p0_bar = np.asarray(self.p0_target, dtype=float).reshape(3) / self.scales.L_ref
        Q0 = np.asarray(self.Q0_target, dtype=float).reshape(4)

        self.C_S_flex = make_C_S_flexible()
        self.C_BV_prox = make_C_BV_proximal_pose(p0_bar, Q0)
        self.C_BV_dist = make_C_BV_distal_free_tip()

        self.M_list = [mesh.M for mesh in self.meshes]

    # ---------- pack/unpack (BAR variables) ----------

    def pack_z(
            self,
            x_nodes_list: List[np.ndarray],
            k_array_list: List[np.ndarray],
            x_rigid_list: List[np.ndarray],
    ) -> np.ndarray:
        pieces = []
        for x_node in x_nodes_list:
            pieces.append(x_node.reshape(-1))
        for k_array in k_array_list:
            pieces.append(k_array.reshape(-1))
        for xR in x_rigid_list:
            pieces.append(xR.reshape(-1))
        return np.concatenate(pieces)

    def unpack_z(self, z: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        cursor = 0
        x_nodes_list: List[np.ndarray] = []
        k_array_list: List[np.ndarray] = []
        x_rigid_list: List[np.ndarray] = []

        for M in self.M_list:
            n_flat = (M + 1) * 13
            block = z[cursor: cursor + n_flat]
            cursor += n_flat
            x_nodes_list.append(block.reshape(M + 1, 13))

        for M in self.M_list:
            k_flat_len = M * 3 * 13
            block = z[cursor: cursor + k_flat_len]
            cursor += k_flat_len
            k_array_list.append(block.reshape(M, 3, 13))

        for _ in range(self.N):
            xR = z[cursor: cursor + 13]
            cursor += 13
            x_rigid_list.append(xR)

        assert cursor == z.size
        return x_nodes_list, k_array_list, x_rigid_list

    # ---------- residual in BAR space ----------

    def residual(self, z: np.ndarray) -> np.ndarray:
        x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar = self.unpack_z(z)
        E_list = []
        s = self.scales

        # ----- flexible segments (GL6 on s_bar) -----
        for i in range(self.N):
            flex = self.flex_segs[i]
            mesh = self.meshes[i]
            x_nodes_bar = x_nodes_list_bar[i]
            k_array_bar = k_array_list_bar[i]

            M = mesh.M
            sbar_nodes = mesh.sbar_nodes
            hbar_list = mesh.hbar_list
            L_seg = float(flex.length)

            for n in range(M):
                x_n_bar = x_nodes_bar[n]
                x_np1_bar = x_nodes_bar[n + 1]
                k_n_bar = k_array_bar[n]
                sbar_n = float(sbar_nodes[n])
                hbar = float(hbar_list[n])

                # boundary constraints for this interval
                if i == 0 and n == 0:
                    def C_BV_flex(xn, xnp1, _fun=self.C_BV_prox):
                        return _fun(xn, xnp1)
                elif i > 0 and n == 0:
                    xR_prev_bar = x_rigid_list_bar[i - 1]

                    def C_BV_flex(xn, xnp1, x_target=xR_prev_bar):
                        return xn - x_target
                else:
                    def C_BV_flex(xn, xnp1):
                        return np.zeros(0)

                # nondim RHS: dx_bar/ds_bar
                def rhs_bar(x_stage_bar: np.ndarray, sbar_stage: float):
                    sigma_dim = float(sbar_stage) * L_seg
                    x_stage_dim = x_bar_to_dim(x_stage_bar, s)
                    dx_dsigma_dim = flex.cosserat_rhs(x_stage_dim, sigma_dim)
                    return rhs_dim_to_rhs_bar_dsbar(dx_dsigma_dim, L_seg=L_seg, s=s)

                E_flex_n = flex.interval_residual_gl6(
                    x_n=x_n_bar,
                    k_n=k_n_bar,
                    x_np1=x_np1_bar,
                    sigma_n=sbar_n,
                    h=hbar,
                    C_BV_fun=C_BV_flex,
                    C_S_fun=self.C_S_flex,
                    rhs_fun=rhs_bar,
                )
                E_list.append(E_flex_n)

        # ----- rigid segments (residual in BAR, but loads in SI) -----
        for i in range(self.N):
            rigid = self.rigid_segs[i]
            x_flex_end_bar = x_nodes_list_bar[i][-1]
            x_rigid_i_bar = x_rigid_list_bar[i]

            if i == self.N - 1:
                def C_BV_rigid(xn, xnp1, _fun=self.C_BV_dist):
                    return _fun(xn, xnp1)
            else:
                def C_BV_rigid(xn, xnp1):
                    return np.zeros(0)

            f_extra = self.f_ext_list[i]
            tau_extra = self.tau_ext_list[i]

            gravity_i = self.gravity_rigid_list[i] if self.gravity_rigid_list is not None else None
            magnet_params_i = self.magnet_params_list[i] if self.magnet_params_list is not None else None

            # convert proximal state to SI for load computation
            x_flex_end_dim = x_bar_to_dim(x_flex_end_bar, s)

            f_auto, tau_auto = compute_external_wrench_total_rigid(
                x_proximal=x_flex_end_dim,
                rigid_length=rigid.length,
                gravity=gravity_i,
                magnetic_model=self.magnetic_model,
                magnet_params=magnet_params_i,
                coil_currents=self.coil_currents,
            )
            f_total = f_extra + f_auto
            tau_total = tau_extra + tau_auto

            # rigid propagation in SI, then back to BAR for residual
            x_rigid_end_dim = rigid.propagate(x_flex_end_dim, f_total, tau_total)
            x_rigid_end_bar = x_dim_to_bar(x_rigid_end_dim, s)

            core = x_rigid_i_bar - x_rigid_end_bar
            Q_np1 = x_rigid_i_bar[3:7]
            quat_norm_res = np.array([np.dot(Q_np1, Q_np1) - 1.0])

            C_BV = C_BV_rigid(x_flex_end_bar, x_rigid_i_bar)
            E_list.append(np.concatenate([core, quat_norm_res, C_BV]))

        return np.concatenate(E_list)

    # ---------- FD Jacobian in BAR space ----------

    def jacobian_fd(self, z: np.ndarray, E: np.ndarray, eps_rel: float = 1e-7, eps_abs: float = 1e-9) -> np.ndarray:
        m = E.size
        n = z.size
        J = np.zeros((m, n))
        for j in range(n):
            zj = z[j]
            eps = max(eps_abs, eps_rel * max(1.0, abs(zj)))
            z_pert = z.copy()
            z_pert[j] = zj + eps
            E_pert = self.residual(z_pert)
            J[:, j] = (E_pert - E) / eps
        return J

    # ---------- LM (works on BAR z) ----------

    def solve(self, z0: np.ndarray, callback: Optional[Callable[[int, np.ndarray, float], None]] = None) -> Tuple[np.ndarray, bool]:
        z = z0.copy()
        lambda_current = self.lm_damping

        E = self.residual(z)
        normE = np.linalg.norm(E)
        cost = 0.5 * normE ** 2
        print(f"[Multi-LM-ND] iter=0, ||E||={normE:.3e}, lambda={lambda_current:.3e}")

        if callback is not None:
            callback(0, z, normE)

        consecutive_accepts = 0

        for it in range(self.max_iter):
            if normE < self.tol:
                print(f"[Multi-LM-ND] converged. Final ||E||={normE:.3e}")
                return z, True

            J = self.jacobian_fd(z, E)
            JtJ = J.T @ J
            g = J.T @ E

            max_inner_iter = 5
            step_accepted = False

            for _ in range(max_inner_iter):
                A = JtJ + lambda_current * np.eye(JtJ.shape[0])
                try:
                    delta = np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    lambda_current *= 10.0
                    continue

                predicted_reduction = float(np.dot(g, delta) - 0.5 * np.dot(delta, JtJ @ delta))
                z_new = z - delta

                E_new = self.residual(z_new)
                normE_new = np.linalg.norm(E_new)
                cost_new = 0.5 * normE_new ** 2
                actual_reduction = cost - cost_new

                rho = actual_reduction / predicted_reduction if abs(predicted_reduction) > 1e-15 else 0.0

                if rho > 0.0:
                    z, E, normE, cost = z_new, E_new, normE_new, cost_new
                    step_accepted = True
                    consecutive_accepts += 1

                    if rho > 0.75:
                        lambda_current *= 0.3
                        status = "↓↓"
                    elif rho > 0.25:
                        lambda_current *= 0.5
                        status = "↓"
                    else:
                        status = "→"

                    lambda_current = max(lambda_current, 1e-12)
                    print(f"[Multi-LM-ND] iter={it+1}, ||E||={normE:.3e}, lambda={lambda_current:.3e} {status}, rho={rho:.2f}")

                    if callback is not None:
                        callback(it + 1, z, normE)
                    break
                else:
                    lambda_current = min(lambda_current * 3.0, 1e10)

            if not step_accepted:
                print(f"[Multi-LM-ND] iter={it+1}, ||E||={normE:.3e}, lambda={lambda_current:.3e} x (Rejected)")
                consecutive_accepts = 0
                if it > 0 and it % 10 == 0:
                    lambda_current = self.lm_damping
                    print(f"[Multi-LM-ND] reset lambda to {lambda_current:.3e}")
            else:
                if consecutive_accepts >= 3:
                    lambda_current = max(lambda_current * 0.5, 1e-12)
                    consecutive_accepts = 0

            # Your debug hook (note: this is called every outer iter)
            self.debug_force_defect_one_interval(z)

        print(f"[Multi-LM-ND] Reached max iterations. Failed. Final ||E||={normE:.3e}")
        return z, False

    # -------------------------------------------------------------------------
    # Rigid force audit identity
    # -------------------------------------------------------------------------

    def audit_rigid_force_identity(self, z_bar: np.ndarray, seg_idx: int = 0) -> None:
        """
        Audit identity (in BAR units):
            f_ext_inferred = f_prox - f_end
        and compare against the computed f_total (converted to BAR).
        We report both:
          - inferred using predicted rigid endpoint from propagate (recommended)
          - inferred using variable rigid endpoint (x_rigid_i_bar) (useful for diagnosis)
        """
        if not self.debug_print:
            return

        x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar = self.unpack_z(z_bar)
        s = self.scales

        rigid = self.rigid_segs[seg_idx]
        x_flex_end_bar = x_nodes_list_bar[seg_idx][-1]
        x_rigid_var_bar = x_rigid_list_bar[seg_idx]

        f_extra = self.f_ext_list[seg_idx]
        tau_extra = self.tau_ext_list[seg_idx]

        gravity_i = self.gravity_rigid_list[seg_idx] if self.gravity_rigid_list is not None else None
        magnet_params_i = self.magnet_params_list[seg_idx] if self.magnet_params_list is not None else None

        # Convert proximal state to SI for load computation
        x_flex_end_dim = x_bar_to_dim(x_flex_end_bar, s)

        f_auto_dim, tau_auto_dim = compute_external_wrench_total_rigid(
            x_proximal=x_flex_end_dim,
            rigid_length=rigid.length,
            gravity=gravity_i,
            magnetic_model=self.magnetic_model,
            magnet_params=magnet_params_i,
            coil_currents=self.coil_currents,
        )
        f_total_dim = f_extra + f_auto_dim
        tau_total_dim = tau_extra + tau_auto_dim

        # Convert loads to BAR for comparison
        f_total_bar = np.asarray(f_total_dim, dtype=float).reshape(3) / s.F_ref
        tau_total_bar = np.asarray(tau_total_dim, dtype=float).reshape(3) / s.M_ref

        # Predicted endpoint from propagation (SI -> BAR)
        x_rigid_end_dim = rigid.propagate(x_flex_end_dim, f_total_dim, tau_total_dim)
        x_rigid_end_bar = x_dim_to_bar(x_rigid_end_dim, s)

        # Forces at proximal and endpoints (BAR)
        f_prox_bar = x_flex_end_bar[7:10]
        f_end_pred_bar = x_rigid_end_bar[7:10]
        f_end_var_bar = x_rigid_var_bar[7:10]

        # Inferred external force (BAR)
        f_ext_inf_pred = f_prox_bar - f_end_pred_bar
        f_ext_inf_var = f_prox_bar - f_end_var_bar

        # Differences vs computed f_total_bar
        diff_pred = f_ext_inf_pred - f_total_bar
        diff_var = f_ext_inf_var - f_total_bar

        def _rel(a: np.ndarray, b: np.ndarray) -> float:
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            den = max(1e-12, nb)
            return na / den

        print(f"\n[RIGID FORCE AUDIT] seg={seg_idx}")
        print(f"  f_prox_bar      = {f_prox_bar}")
        print(f"  f_end_pred_bar  = {f_end_pred_bar}")
        print(f"  f_end_var_bar   = {f_end_var_bar}")
        print(f"  f_total_dim (SI)= {np.asarray(f_total_dim).reshape(3)}")
        print(f"  f_total_bar     = {f_total_bar}   ||f_total_bar||={np.linalg.norm(f_total_bar):.6g}")
        print(f"  inferred_pred   = {f_ext_inf_pred} ||·||={np.linalg.norm(f_ext_inf_pred):.6g}")
        print(f"  inferred_var    = {f_ext_inf_var}  ||·||={np.linalg.norm(f_ext_inf_var):.6g}")
        print(f"  diff_pred       = {diff_pred}      ||·||={np.linalg.norm(diff_pred):.6g}  rel={_rel(diff_pred, f_total_bar):.6g}")
        print(f"  diff_var        = {diff_var}       ||·||={np.linalg.norm(diff_var):.6g}  rel={_rel(diff_var, f_total_bar):.6g}")

        # Optional: also show torque totals (often helps, but not part of force identity)
        print(f"  tau_total_dim(SI)= {np.asarray(tau_total_dim).reshape(3)}")
        print(f"  tau_total_bar    = {tau_total_bar}   ||tau_total_bar||={np.linalg.norm(tau_total_bar):.6g}")

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def debug_force_defect_one_interval(self, z_bar, seg_idx=0, interval_idx=2):
        """
        Debug flex force defect for one interval + rigid force audit.
        """
        if not self.debug_print:
            return

        x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar = self.unpack_z(z_bar)

        # # sanity: right endpoint indexing
        # try:
        #     x_end_from_nodes = x_nodes_list_bar[seg_idx][-1]
        #     x_np1_interval2 = x_nodes_list_bar[seg_idx][interval_idx + 1]
        #     print("||x_end_from_nodes - x_np1_interval|| =",
        #           np.linalg.norm(x_end_from_nodes - x_np1_interval2))
        # except Exception:
        #     pass
        #
        # # quick compare flex_end vs rigid variable (not an error by itself)
        # x_flex_end = x_nodes_list_bar[seg_idx][-1]
        # x_rigid_var = x_rigid_list_bar[seg_idx]
        # print("flex_end f =", x_flex_end[7:10])
        # print("rigidVar f =", x_rigid_var[7:10])
        # print("||flex_end - rigidVar|| =", np.linalg.norm(x_flex_end - x_rigid_var))

        # rigid force audit identity
        # if self.debug_audit_rigid_force and seg_idx == self.debug_audit_seg_idx:
        #     self.audit_rigid_force_identity(z_bar, seg_idx=seg_idx)

        flex = self.flex_segs[seg_idx]
        mesh = self.meshes[seg_idx]
        s = self.scales

        # interval endpoints
        x_n = x_nodes_list_bar[seg_idx][interval_idx].copy()
        x_np1 = x_nodes_list_bar[seg_idx][interval_idx + 1].copy()
        k_n = k_array_list_bar[seg_idx][interval_idx].copy()  # (3,13)

        # use the same nondim mesh variables as residual()
        h = float(mesh.hbar_list[interval_idx])
        sigma_n = float(mesh.sbar_nodes[interval_idx])

        def rhs_bar(x_stage_bar, sigma_stage_bar):
            sigma_dim = float(sigma_stage_bar) * float(flex.length)
            x_stage_dim = x_bar_to_dim(x_stage_bar, s)
            dx_dsigma_dim = flex.cosserat_rhs(x_stage_dim, sigma_dim)
            return rhs_dim_to_rhs_bar_dsbar(dx_dsigma_dim, L_seg=flex.length, s=s)

        # compute stage states and rhs
        xs = []
        gs = []
        for i in range(3):
            x_stage = x_n.copy()
            for j in range(3):
                x_stage = x_stage + h * GL3_A[i, j] * k_n[j]
            sigma_stage = sigma_n + GL3_C[i] * h
            g_i = rhs_bar(x_stage, sigma_stage)

            xs.append(x_stage)
            gs.append(g_i)

        gs = np.stack(gs, axis=0)  # (3,13)

        # f-related pieces
        f_n = x_n[7:10]
        f_np1 = x_np1[7:10]
        df_node = f_np1 - f_n

        kf = k_n[:, 7:10]  # (3,3)
        df_pred = h * (GL3_B @ kf)  # (3,)
        defect_f = df_node - df_pred

        gk_f = gs[:, 7:10]  # rhs df/dsbar

        # print(f"\n[debug seg={seg_idx} interval={interval_idx}]")
        # print(f"  f_n    = {f_n}")
        # print(f"  f_np1  = {f_np1}")
        # print(f"  df_node     = {df_node}   ||df_node||={np.linalg.norm(df_node):.6g}")
        # print(f"  k_f(stages) =\n{kf}")
        # print(f"  rhs_f(stages)=\n{gk_f}")
        # print(f"  df_pred(GL) = {df_pred}   ||df_pred||={np.linalg.norm(df_pred):.6g}")
        # print(f"  defect_f    = {defect_f}  ||defect_f||={np.linalg.norm(defect_f):.6g}")
        #
        # if flex.fext_density is not None:
        #     for i in range(3):
        #         sigma_dim = float((sigma_n + GL3_C[i] * h) * flex.length)
        #         x_stage_dim = x_bar_to_dim(xs[i], s)
        #         fb = flex.fext_density(x_stage_dim, sigma_dim)
        #         print(f"  fext_density(stage {i}) = {fb}")
        # else:
        #     print("  flex.fext_density is None (good)")

    # --- (下面保留你已有的 residual_block_report / print_top_blocks_verbose / print_defect_components 原样) ---

    def residual_block_report(self, z: np.ndarray, cs_min_len: int = 2) -> dict:
        x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar = self.unpack_z(z)
        s = self.scales

        report = {"flex": [], "rigid": [], "total_norm": None}

        for i in range(self.N):
            flex = self.flex_segs[i]
            mesh = self.meshes[i]
            x_nodes_bar = x_nodes_list_bar[i]
            k_array_bar = k_array_list_bar[i]

            M = mesh.M
            sbar_nodes = mesh.sbar_nodes
            hbar_list = mesh.hbar_list
            L_seg = float(flex.length)

            for n in range(M):
                x_n_bar = x_nodes_bar[n]
                x_np1_bar = x_nodes_bar[n + 1]
                k_n_bar = k_array_bar[n]
                sbar_n = float(sbar_nodes[n])
                hbar = float(hbar_list[n])

                if i == 0 and n == 0:
                    def C_BV_flex(xn, xnp1, _fun=self.C_BV_prox):
                        return _fun(xn, xnp1)
                elif i > 0 and n == 0:
                    xR_prev_bar = x_rigid_list_bar[i - 1]

                    def C_BV_flex(xn, xnp1, x_target=xR_prev_bar):
                        return xn - x_target
                else:
                    def C_BV_flex(xn, xnp1):
                        return np.zeros(0)

                def rhs_bar(x_stage_bar: np.ndarray, sbar_stage: float):
                    sigma_dim = float(sbar_stage) * L_seg
                    x_stage_dim = x_bar_to_dim(x_stage_bar, s)
                    dx_dsigma_dim = flex.cosserat_rhs(x_stage_dim, sigma_dim)
                    return rhs_dim_to_rhs_bar_dsbar(dx_dsigma_dim, L_seg=L_seg, s=s)

                E_flex_n = flex.interval_residual_gl6(
                    x_n=x_n_bar,
                    k_n=k_n_bar,
                    x_np1=x_np1_bar,
                    sigma_n=sbar_n,
                    h=hbar,
                    C_BV_fun=C_BV_flex,
                    C_S_fun=self.C_S_flex,
                    rhs_fun=rhs_bar,
                )

                E = np.asarray(E_flex_n, dtype=float).ravel()
                L = E.size

                cs_len = int(cs_min_len)
                if L < cs_len + 13 + 39:
                    report["flex"].append({
                        "seg": i, "interval": n,
                        "norm": float(np.linalg.norm(E)),
                        "maxabs": float(np.max(np.abs(E))) if L else 0.0,
                        "len": int(L),
                        "sub": {"parse_ok": False, "reason": "len too small for expected sub-blocks"}
                    })
                    continue

                idx0 = 0
                idx1 = cs_len
                idx2 = idx1 + 13
                idx3 = idx2 + 39

                C_S = E[idx0:idx1]
                res_state = E[idx1:idx2]
                res_ks = E[idx2:idx3]
                C_BV = E[idx3:]

                ks_mat = res_ks.reshape(3, 13)
                ks_stage_norms = [float(np.linalg.norm(ks_mat[ii])) for ii in range(3)]
                ks_stage_maxabs = [float(np.max(np.abs(ks_mat[ii]))) for ii in range(3)]

                report["flex"].append({
                    "seg": i,
                    "interval": n,
                    "norm": float(np.linalg.norm(E)),
                    "maxabs": float(np.max(np.abs(E))) if L else 0.0,
                    "len": int(L),
                    "sub": {
                        "parse_ok": True,
                        "cs_len": int(cs_len),
                        "bv_len": int(C_BV.size),

                        "C_S_norm": float(np.linalg.norm(C_S)),
                        "C_S_maxabs": float(np.max(np.abs(C_S))) if C_S.size else 0.0,

                        "state_norm": float(np.linalg.norm(res_state)),
                        "state_maxabs": float(np.max(np.abs(res_state))) if res_state.size else 0.0,

                        "ks_norm": float(np.linalg.norm(res_ks)),
                        "ks_maxabs": float(np.max(np.abs(res_ks))) if res_ks.size else 0.0,
                        "ks_stage_norms": ks_stage_norms,
                        "ks_stage_maxabs": ks_stage_maxabs,

                        "BV_norm": float(np.linalg.norm(C_BV)) if C_BV.size else 0.0,
                        "BV_maxabs": float(np.max(np.abs(C_BV))) if C_BV.size else 0.0,
                    }
                })

        for i in range(self.N):
            rigid = self.rigid_segs[i]
            x_flex_end_bar = x_nodes_list_bar[i][-1]
            x_rigid_i_bar = x_rigid_list_bar[i]

            if i == self.N - 1:
                def C_BV_rigid(xn, xnp1, _fun=self.C_BV_dist):
                    return _fun(xn, xnp1)
            else:
                def C_BV_rigid(xn, xnp1):
                    return np.zeros(0)

            f_extra = self.f_ext_list[i]
            tau_extra = self.tau_ext_list[i]
            gravity_i = self.gravity_rigid_list[i] if self.gravity_rigid_list is not None else None
            magnet_params_i = self.magnet_params_list[i] if self.magnet_params_list is not None else None

            x_flex_end_dim = x_bar_to_dim(x_flex_end_bar, s)

            f_auto, tau_auto = compute_external_wrench_total_rigid(
                x_proximal=x_flex_end_dim,
                rigid_length=rigid.length,
                gravity=gravity_i,
                magnetic_model=self.magnetic_model,
                magnet_params=magnet_params_i,
                coil_currents=self.coil_currents,
            )

            f_total = f_extra + f_auto
            tau_total = tau_extra + tau_auto

            x_rigid_end_dim = rigid.propagate(x_flex_end_dim, f_total, tau_total)
            x_rigid_end_bar = x_dim_to_bar(x_rigid_end_dim, s)

            core = x_rigid_i_bar - x_rigid_end_bar
            Q_np1 = x_rigid_i_bar[3:7]
            quat_norm_res = np.array([np.dot(Q_np1, Q_np1) - 1.0])
            C_BV = C_BV_rigid(x_flex_end_bar, x_rigid_i_bar)

            E_rigid = np.concatenate([core, quat_norm_res, C_BV])
            report["rigid"].append({
                "seg": i,
                "norm": float(np.linalg.norm(E_rigid)),
                "maxabs": float(np.max(np.abs(E_rigid))) if E_rigid.size else 0.0,
                "len": int(E_rigid.size),
            })

        E_all = self.residual(z)
        report["total_norm"] = float(np.linalg.norm(E_all))
        return report

    def print_top_blocks_verbose(self, rep: dict, topk: int = 8):
        print("total", rep["total_norm"])

        flex_sorted = sorted(rep["flex"], key=lambda d: d["norm"], reverse=True)
        print("\nTop flex blocks (verbose):")
        for b in flex_sorted[:topk]:
            sub = b.get("sub", {})
            if not sub.get("parse_ok", False):
                print(b)
                continue

            print(
                f"  seg={b['seg']} interval={b['interval']} len={b['len']} "
                f"||E||={b['norm']:.6g} max={b['maxabs']:.6g} | "
                f"C_S={sub['C_S_norm']:.3g}, state={sub['state_norm']:.3g}, "
                f"ks={sub['ks_norm']:.3g} (stages {sub['ks_stage_norms']}), "
                f"BV={sub['BV_norm']:.3g} (bv_len={sub['bv_len']})"
            )

        rigid_sorted = sorted(rep["rigid"], key=lambda d: d["norm"], reverse=True)
        print("\nRigid blocks:")
        for b in rigid_sorted:
            print(
                f"  seg={b['seg']} len={b['len']} ||E||={b['norm']:.6g} max={b['maxabs']:.6g}"
            )

    def print_defect_components(self, z, seg_id, interval_id, cs_len=2):
        rep = self.residual_block_report(z, cs_min_len=cs_len)
        b = None
        for bb in rep["flex"]:
            if bb["seg"] == seg_id and bb["interval"] == interval_id:
                b = bb
                break
        if b is None or not b.get("sub", {}).get("parse_ok", False):
            print("block not found or parse failed")
            return

        x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar = self.unpack_z(z)
        flex = self.flex_segs[seg_id]
        mesh = self.meshes[seg_id]
        s = self.scales
        L_seg = float(flex.length)

        x_n = x_nodes_list_bar[seg_id][interval_id]
        x_np1 = x_nodes_list_bar[seg_id][interval_id + 1]
        k_n = k_array_list_bar[seg_id][interval_id]
        sigma_n = float(mesh.sbar_nodes[interval_id])
        h = float(mesh.hbar_list[interval_id])

        if seg_id == 0 and interval_id == 0:
            def C_BV_flex(xn, xnp1, _fun=self.C_BV_prox): return _fun(xn, xnp1)
        elif seg_id > 0 and interval_id == 0:
            xR_prev = x_rigid_list_bar[seg_id - 1]

            def C_BV_flex(xn, xnp1, x_target=xR_prev): return xn - x_target
        else:
            def C_BV_flex(xn, xnp1): return np.zeros(0)

        def rhs_bar(x_stage_bar, sbar_stage):
            sigma_dim = float(sbar_stage) * L_seg
            x_stage_dim = x_bar_to_dim(x_stage_bar, s)
            dx_dsigma_dim = flex.cosserat_rhs(x_stage_dim, sigma_dim)
            return rhs_dim_to_rhs_bar_dsbar(dx_dsigma_dim, L_seg=L_seg, s=s)

        E = flex.interval_residual_gl6(
            x_n=x_n, k_n=k_n, x_np1=x_np1,
            sigma_n=sigma_n, h=h,
            C_BV_fun=C_BV_flex, C_S_fun=self.C_S_flex, rhs_fun=rhs_bar
        ).ravel()

        res_state = E[cs_len:cs_len + 13]
        p_def = res_state[0:3]
        Q_def = res_state[3:7]
        f_def = res_state[7:10]
        tau_def = res_state[10:13]

        print(f"[defect seg={seg_id} interval={interval_id}] ||state||={np.linalg.norm(res_state):.6g}")
        print(f"  p  norm={np.linalg.norm(p_def):.6g}, max={np.max(np.abs(p_def)):.6g}, vec={p_def}")
        print(f"  Q  norm={np.linalg.norm(Q_def):.6g}, max={np.max(np.abs(Q_def)):.6g}, vec={Q_def}")
        print(f"  f  norm={np.linalg.norm(f_def):.6g}, max={np.max(np.abs(f_def)):.6g}, vec={f_def}")
        print(f"  tau norm={np.linalg.norm(tau_def):.6g}, max={np.max(np.abs(tau_def)):.6g}, vec={tau_def}")
