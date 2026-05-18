from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from .basics import (
    skew,
    quat_to_rotmat,
    quat_derivative,
    GL3_A,
    GL3_B,
    GL3_C,
)



@dataclass
class Segment:
    """Base class for flexible and rigid catheter segments."""
    length: float
    name: str = ""

    def propagate(self, *args, **kwargs):
        """Propagate the segment state along its centerline."""
        raise NotImplementedError




@dataclass
class FlexibleSegment(Segment):
    """Cosserat-rod flexible segment."""
    K_se: np.ndarray = np.eye(3)
    K_bt: np.ndarray = np.eye(3)
    v_star: np.ndarray = np.array([0., 0., 1.])
    u_star: np.ndarray = np.array([0., 0., 0.])

    fext_density: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    tauext_density: Optional[Callable[[np.ndarray, float], np.ndarray]] = None


    def cosserat_rhs(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Evaluate the Cosserat rod right-hand side."""

        p = x[:3]
        Q = x[3:7]
        f = x[7:10]
        tau = x[10:13]

        Q = Q / np.linalg.norm(Q)

        R = quat_to_rotmat(Q)

        if self.fext_density is None:
            f_bar = np.zeros(3)
        else:
            f_bar = self.fext_density(x, sigma)

        if self.tauext_density is None:
            tau_bar = np.zeros(3)
        else:
            tau_bar = self.tauext_density(x, sigma)

        Kse_inv = np.linalg.inv(self.K_se)
        Kbt_inv = np.linalg.inv(self.K_bt)

        v = Kse_inv @ (R.T @ f) + self.v_star
        u = Kbt_inv @ (R.T @ tau) + self.u_star

        # dp/dsigma = R(Q) * v
        dp_dsigma = R @ v

        # dQ/dsigma = 1/2 * [ -q^T; qr I - [q]_x ] * omega
        omega_global = R @ u
        dQ_dsigma = quat_derivative(Q, omega_global)

        # df/dsigma = f_bar
        df_dsigma = - f_bar

        # dtau/dsigma = [f]_x dp/dsigma - \bar τ_ext(x, σ)
        df_x = skew(f)
        dtau_dsigma = df_x @ dp_dsigma - tau_bar

        dx_sigma = np.zeros_like(x)
        dx_sigma[:3] = dp_dsigma
        dx_sigma[3:7] = dQ_dsigma
        dx_sigma[7:10] = df_dsigma
        dx_sigma[10:13] = dtau_dsigma

        return dx_sigma


    def propagate(self,
                  x_proximal: np.ndarray,
                  integrator: Callable[[Callable, np.ndarray, float, float], np.ndarray],
                  n_steps: int = 1) -> np.ndarray:
        """Propagate the flexible segment from proximal to distal end."""

        x = x_proximal.copy()
        s0 = 0.
        s1 = self.length
        if n_steps <= 0:
            n_steps = 1
        h = (s1 - s0) / n_steps

        sigma = s0
        for _ in range(n_steps):
            x = integrator(self.cosserat_rhs, x, sigma, sigma + h)
            sigma += h

        return x


    def interval_residual_gl6(
            self,
            x_n: np.ndarray,
            k_n: np.ndarray,
            x_np1: np.ndarray,
            sigma_n: float,
            h: float,
            C_BV_fun=None,
            C_S_fun=None,
    ) -> np.ndarray:
        """Assemble the GL6 residual for one flexible interval."""

        if C_S_fun is None:
            C_S = np.zeros(0)
        else:
            C_S = C_S_fun(x_n, x_np1)

        res_state = x_np1 - x_n - h * sum(
            GL3_B[i] * k_n[i] for i in range(3)
        )

        res_ks = []
        for i in range(3):
            # x_stage_i = x_n + h Σ_j a_ij k_j
            x_stage_i = x_n.copy()
            for j in range(3):
                x_stage_i = x_stage_i + h * GL3_A[i, j] * k_n[j]

            sigma_stage_i = sigma_n + GL3_C[i] * h

            # g(x, σ) = cosserat_rhs(x, σ)
            g_i = self.cosserat_rhs(x_stage_i, sigma_stage_i)

            res_ki = k_n[i] - g_i
            res_ks.append(res_ki)

        res_ks = np.concatenate(res_ks, axis=0)

        if C_BV_fun is None:
            C_BV = np.zeros(0)
        else:
            C_BV = C_BV_fun(x_n, x_np1)

        E_n = np.concatenate([C_S, res_state, res_ks, C_BV])
        return E_n




@dataclass
class RigidSegment(Segment):
    """Rigid magnet segment with an analytic propagation model."""

    v_star: np.ndarray = np.array([0., 0., 1.])
    f_ext: np.ndarray = np.zeros(3)
    tau_ext: np.ndarray = np.zeros(3)

    def state_along(
            self,
            x_proximal: np.ndarray,
            sigma: float,
            f_ext_total: np.ndarray,
            tau_ext_total: np.ndarray
    ) -> np.ndarray:
        """Evaluate the rigid-segment state at a given arc-length coordinate."""

        L = self.length

        sigma = float(np.clip(sigma, 0., L))

        pp = x_proximal[:3]
        Qp = x_proximal[3:7]
        fp = x_proximal[7:10]
        tau_p = x_proximal[10:13]

        Qp = Qp / np.linalg.norm(Qp)
        R_p = quat_to_rotmat(Qp)

        p_sigma = pp + sigma * (R_p @ self.v_star)
        Q_sigma = Qp.copy()

        f_sigma = fp - (sigma / L) * f_ext_total

        r = p_sigma - pp
        term1 = tau_p - (sigma / L) * tau_ext_total
        term2 = - skew(r) @ fp
        term3 = (sigma / (2. * L)) * (skew(r) @ f_ext_total)
        tau_sigma = term1 + term2 + term3

        x_sigma = np.zeros_like(x_proximal)
        x_sigma[:3] = p_sigma
        x_sigma[3:7] = Q_sigma
        x_sigma[7:10] = f_sigma
        x_sigma[10:13] = tau_sigma

        return x_sigma

    def propagate(
            self,
            x_proximal: np.ndarray,
            f_ext_total: np.ndarray,
            tau_ext_total: np.ndarray,
    ) -> np.ndarray:
        """Return the rigid-segment distal state."""
        return self.state_along(
            x_proximal=x_proximal,
            sigma=self.length,
            f_ext_total=f_ext_total,
            tau_ext_total=tau_ext_total,
        )

    def interval_residual_rigid(
            self,
            x_n: np.ndarray,
            x_np1: np.ndarray,
            f_ext_total: np.ndarray,
            tau_ext_total: np.ndarray,
            C_BV_fun=None,
            C_S_fun=None,
    ) -> np.ndarray:
        """Assemble the residual for a rigid segment block."""
        x_rigid_end = self.propagate(x_n, f_ext_total, tau_ext_total)

        C_S_core = x_np1 - x_rigid_end
        Q_np1 = x_np1[3:7]
        quat_norm_res = np.array([np.dot(Q_np1, Q_np1) - 1.])

        C_S_list = [C_S_core, quat_norm_res]

        if C_S_fun is not None:
            C_S_list.append(C_S_fun(x_n, x_np1))

        C_S = np.concatenate(C_S_list)

        if C_BV_fun is None:
            C_BV = np.zeros(0)
        else:
            C_BV = C_BV_fun(x_n, x_np1)

        E_n = np.concatenate([C_S, C_BV])

        return E_n
