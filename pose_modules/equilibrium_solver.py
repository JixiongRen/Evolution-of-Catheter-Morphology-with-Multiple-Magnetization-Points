import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
from .segments import RigidSegment, FlexibleSegment
from .rod_mesh import RodMesh
from .utils import make_C_S_flexible, make_C_BV_proximal_pose, make_C_BV_distal_free_tip
from .external_wrench import GravityRigid, MagneticModel, compute_external_wrench_total_rigid


@dataclass
class MultiSegmentEquilibriumSolver:
    """Solve static equilibrium for a multi-segment catheter."""

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

    max_iter: int = 100000
    tol: float = 1e-6
    lm_damping: float = 1e-3


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

        self.C_S_flex = make_C_S_flexible()
        self.C_BV_prox = make_C_BV_proximal_pose(self.p0_target, self.Q0_target)
        self.C_BV_dist = make_C_BV_distal_free_tip()

        self.M_list = [mesh.M for mesh in self.meshes]


    def pack_z(
            self,
            x_nodes_list: List[np.ndarray],
            k_array_list: List[np.ndarray],
            x_rigid_list: List[np.ndarray],
    ) -> np.ndarray:
        """Pack solver blocks into a single optimization vector."""
        pieces = []

        for x_node in x_nodes_list:
            pieces.append(x_node.reshape(-1))
        for k_array in k_array_list:
            pieces.append(k_array.reshape(-1))
        for xR in x_rigid_list:
            pieces.append(xR.reshape(-1))

        z = np.concatenate(pieces)
        return z


    def unpack_z(
            self,
            z: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Unpack the optimization vector into solver blocks."""
        cursor = 0
        x_nodes_list: List[np.ndarray] = []
        k_array_list: List[np.ndarray] = []
        x_rigid_list: List[np.ndarray] = []

        for i, M in enumerate(self.M_list):
            n_flat = (M + 1) * 13
            block = z[cursor: cursor + n_flat]
            cursor += n_flat
            x_nodes = block.reshape(M + 1, 13)
            x_nodes_list.append(x_nodes)


        for i, M in enumerate(self.M_list):
            k_flat_len = M * 3 * 13
            block = z[cursor: cursor + k_flat_len]
            cursor += k_flat_len
            k_array = block.reshape(M, 3, 13)
            k_array_list.append(k_array)


        for _ in range(self.N):
            xR = z[cursor: cursor + 13]
            cursor += 13
            x_rigid_list.append(xR)


        assert cursor == z.size
        return x_nodes_list, k_array_list, x_rigid_list




    def residual(self, z: np.ndarray) -> np.ndarray:
        """Assemble the global residual vector."""
        x_nodes_list, k_array_list, x_rigid_list = self.unpack_z(z)
        E_list = []

        for i in range(self.N):
            flex = self.flex_segs[i]
            mesh = self.meshes[i]
            x_nodes = x_nodes_list[i]
            k_array = k_array_list[i]
            M = mesh.M
            sigmas = mesh.sigma_nodes
            h_list = mesh.h_list

            for n in range(M):
                x_n = x_nodes[n]
                x_np1 = x_nodes[n + 1]
                k_n = k_array[n]
                sigma_n = float(sigmas[n])
                h = float(h_list[n])

                # C_BV_flex
                if i == 0 and n == 0:
                    def C_BV_flex(xn, xnp1, _fun=self.C_BV_prox):
                        return _fun(xn, xnp1)
                elif i > 0 and n == 0:
                    xR_prev = x_rigid_list[i - 1]

                    def C_BV_flex(xn, xnp1, x_target=xR_prev):
                        return xn - x_target
                else:
                    def C_BV_flex(xn, xnp1):
                        return np.zeros(0)

                E_flex_n = flex.interval_residual_gl6(
                    x_n=x_n,
                    k_n=k_n,
                    x_np1=x_np1,
                    sigma_n=sigma_n,
                    h=h,
                    C_BV_fun=C_BV_flex,
                    C_S_fun=self.C_S_flex,
                )
                E_list.append(E_flex_n)

        for i in range(self.N):
            rigid = self.rigid_segs[i]
            x_flex_end_i = x_nodes_list[i][-1]
            x_rigid_i = x_rigid_list[i]


            if i == self.N - 1:
                def C_BV_rigid(xn, xnp1, _fun=self.C_BV_dist):
                    return _fun(xn, xnp1)
            else:
                def C_BV_rigid(xn, xnp1):
                    return np.zeros(0)

            f_extra = self.f_ext_list[i]
            tau_extra = self.tau_ext_list[i]

            gravity_i = None
            if self.gravity_rigid_list is not None:
                gravity_i = self.gravity_rigid_list[i]

            magnet_params_i = None
            if self.magnet_params_list is not None:
                magnet_params_i = self.magnet_params_list[i]

            f_auto, tau_auto = compute_external_wrench_total_rigid(
                x_proximal=x_flex_end_i,
                rigid_length=rigid.length,
                gravity=gravity_i,
                magnetic_model=self.magnetic_model,
                magnet_params=magnet_params_i,
                coil_currents=self.coil_currents,
            )

            f_total = f_extra + f_auto
            tau_total = tau_extra + tau_auto

            E_rigid_i = rigid.interval_residual_rigid(
                x_n=x_flex_end_i,
                x_np1=x_rigid_i,
                f_ext_total=f_total,
                tau_ext_total=tau_total,
                C_BV_fun=C_BV_rigid,
                C_S_fun=None,
            )
            E_list.append(E_rigid_i)

        return np.concatenate(E_list)




    def jacobian_fd(self, z: np.ndarray, E: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Approximate the Jacobian with forward finite differences."""
        m = E.size
        n = z.size
        J = np.zeros((m, n))
        for j in range(n):
            z_pert = z.copy()
            z_pert[j] += eps
            E_pert = self.residual(z_pert)
            J[:, j] = (E_pert - E) / eps
        return J


    # ------------- Levenberg-Marquardt Algorithm -------------


    def solve(self, z0: np.ndarray, callback: Optional[Callable[[int, np.ndarray, float], None]] = None) -> Tuple[np.ndarray, bool]:
        """

        :param z0:
        :param callback:
        :return:
        """

        z = z0.copy()
        lambda_current = self.lm_damping

        E = self.residual(z)
        normE = np.linalg.norm(E)
        cost = 0.5 * normE**2
        print(f"[Multi-LM] iter=0, ||E||={normE:.3e}, lambda={lambda_current:.3e}")

        if callback is not None:
            callback(0, z, normE)

        consecutive_accepts = 0

        _quit_event = None
        try:
            import sys as _sys, threading as _threading
            if _sys.stdin and hasattr(_sys.stdin, "readable") and _sys.stdin.readable():
                _quit_event = _threading.Event()
                def _listener():
                    try:
                        while not _quit_event.is_set():
                            ch = _sys.stdin.read(1)
                            if not ch:
                                break
                            if ch in ('q', 'Q'):
                                _quit_event.set()
                                break
                    except Exception:
                        pass
                _t = _threading.Thread(target=_listener, daemon=True)
                _t.start()
        except Exception:
            pass

        for it in range(self.max_iter):
            if _quit_event is not None and _quit_event.is_set():
                print("[Multi-LM] user quit")
                return z, False
            if normE < self.tol:
                print(f"[Multi-LM] converged successfully. Final ||E||={normE:.3e}")
                return z, True

            J = self.jacobian_fd(z, E)
            JtJ = J.T @ J
            g = J.T @ E

            max_inner_iter = 5
            step_accepted = False

            for inner_it in range(max_inner_iter):
                A = JtJ + lambda_current * np.eye(JtJ.shape[0])
                try:
                    delta = np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    lambda_current *= 10.0
                    continue

                predicted_reduction = np.dot(g, delta) - 0.5 * np.dot(delta, JtJ @ delta)

                z_new = z - delta
                E_new = self.residual(z_new)
                normE_new = np.linalg.norm(E_new)
                cost_new = 0.5 * normE_new ** 2

                actual_reduction = cost - cost_new

                if abs(predicted_reduction) > 1e-15:
                    rho = actual_reduction / predicted_reduction
                else:
                    rho = 0.0

                if rho > 0.0:
                    z = z_new
                    E = E_new
                    normE = normE_new
                    cost = cost_new
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
                    print(
                        f"[Multi-LM] iter={it + 1}, ||E||={normE:.3e}, lambda={lambda_current:.3e} {status}, rho={rho:.2f}")

                    if callback is not None:
                        callback(it + 1, z, normE)

                    break
                else:
                    lambda_current *= 3.0
                    lambda_current = min(lambda_current, 1e10)

            if not step_accepted:
                print(f"[Multi-LM] iter={it + 1}, ||E||={normE:.3e}, lambda={lambda_current:.3e} x (Rejected)")
                consecutive_accepts = 0

                if it > 0 and it % 10 == 0:
                    lambda_current = self.lm_damping
                    print(f"[Multi-LM] reset lambda to {lambda_current:.3e}")
            else:
                if consecutive_accepts >= 3:
                    lambda_current *= 0.5
                    lambda_current = max(lambda_current, 1e-12)
                    consecutive_accepts = 0

        print(f"[Multi-LM] Reached max iterations. Converge failed. Final ||E||={normE:.3e}")
        return z, False

















