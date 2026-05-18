from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Protocol, Literal
import numpy as np
from .segments import quat_to_rotmat




@dataclass
class GravityLineDensity:
    """Distributed gravity load for a flexible segment."""
    rhoA: float
    g_vec: np.ndarray


    def force_world(self) -> np.ndarray:
        v = np.asarray(self.g_vec, dtype=float).reshape(3)
        return self.rhoA * v


@dataclass
class GravityRigid:
    """Gravity model for a rigid segment."""
    mass: float
    g_world: np.ndarray
    r_cm_body: np.ndarray


    def total_force_world(self) -> np.ndarray:
        # G = m * g
        return self.mass * self.g_world.reshape(3)

    def total_torque_body(self) -> np.ndarray:
        """Raise because rigid torque is evaluated in the world frame."""
        raise NotImplementedError("Use world-frame torque computation with segment pose.")




class MagneticModel(Protocol):
    """Protocol for magnetic wrench models."""

    def wrench_on_magnet(
            self,
            position_world: np.ndarray,
            R_world_from_body: np.ndarray,
            magnet_params: dict,
            coil_currents: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the magnetic force and torque in the world frame."""
        ...




# def make_external_wrench_density_flexible(
#         gravity: Optional[GravityLineDensity] = None,
#         magnetic_density_fun: Optional[
#             Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]
#         ] = None,
# ):
#     """
#
#     ----
#     gravity : GravityLineDensity or None
#     magnetic_density_fun : callable or None
#     """
#
#     def fext_density(x: np.ndarray, sigma: float) -> np.ndarray:
#         """
#         """
#
#         Q = x[3:7]
#         Q = Q / np.linalg.norm(Q)
#         R = quat_to_rotmat(Q)
#
#         f_total_body = np.zeros(3)
#
#         if gravity is not None:
#             f_g_world = gravity.force_world()  # (3,)
#             f_g_body = R.T @ f_g_world
#             f_total_body += f_g_body
#
#         if magnetic_density_fun is not None:
#             f_mag_body, _ = magnetic_density_fun(x, sigma)
#             f_total_body += f_mag_body
#
#         return f_total_body
#
#
#     def tauext_density(x: np.ndarray, sigma: float) -> np.ndarray:
#         """
#         """
#
#         tau_total_body = np.zeros(3)
#
#
#         if magnetic_density_fun is not None:
#             _, tau_mag_body = magnetic_density_fun(x, sigma)
#             tau_total_body += tau_mag_body
#
#         return tau_total_body
#
#     return fext_density, tauext_density



def make_external_wrench_density_flexible(
        gravity: Optional[GravityLineDensity] = None,
        magnetic_density_fun: Optional[
            Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]
        ] = None,
):
    """Build distributed load callbacks for a flexible segment."""

    def fext_density(x: np.ndarray, sigma: float) -> np.ndarray:
        Q = x[3:7]
        Q = Q / np.linalg.norm(Q)
        R = quat_to_rotmat(Q)  # world_from_body

        f_total_world = np.zeros(3)

        if gravity is not None:
            f_total_world += gravity.force_world()  # (3,) world

        if magnetic_density_fun is not None:
            f_mag_body, _ = magnetic_density_fun(x, sigma)
            f_total_world += R @ np.asarray(f_mag_body, dtype=float).reshape(3)

        return f_total_world

    def tauext_density(x: np.ndarray, sigma: float) -> np.ndarray:
        Q = x[3:7]
        Q = Q / np.linalg.norm(Q)
        R = quat_to_rotmat(Q)  # world_from_body

        tau_total_world = np.zeros(3)


        if magnetic_density_fun is not None:
            _, tau_mag_body = magnetic_density_fun(x, sigma)
            tau_total_world += R @ np.asarray(tau_mag_body, dtype=float).reshape(3)

        return tau_total_world

    return fext_density, tauext_density



def compute_external_wrench_total_rigid(
        x_proximal: np.ndarray,
        rigid_length: float,
        gravity: Optional[GravityRigid] = None,
        magnetic_model: Optional[MagneticModel] = None,
        magnet_params: Optional[dict] = None,
        coil_currents: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the total world-frame wrench on a rigid segment."""

    p0 = x_proximal[:3]
    Q0 = x_proximal[3:7]
    Q0 = Q0 / np.linalg.norm(Q0)
    R0 = quat_to_rotmat(Q0)  # world_from_body

    f_total_world = np.zeros(3)
    tau_total_world = np.zeros(3)

    if gravity is not None:
        Fg_world = gravity.total_force_world()  # (3,)
        r_cm_body = gravity.r_cm_body.reshape(3)
        r_cm_world = R0 @ r_cm_body + p0  # position of CM in world
        r_vec = r_cm_world - p0
        tau_g_world = np.cross(r_vec, Fg_world)

        f_total_world += Fg_world
        tau_total_world += tau_g_world

    if magnetic_model is not None and magnet_params is not None and coil_currents is not None:
        r_mid_body = np.array([0.0, 0.0, rigid_length / 2.0])
        p_mid_world = R0 @ r_mid_body + p0

        Fm_world, Tau_world_about_mid = magnetic_model.wrench_on_magnet(
            position_world=p_mid_world,
            R_world_from_body=R0,
            magnet_params=magnet_params,
            coil_currents=coil_currents,
        )

        r_mid = p_mid_world - p0
        tau_m_world = np.cross(r_mid, Fm_world) + Tau_world_about_mid

        f_total_world += Fm_world
        tau_total_world += tau_m_world

    return f_total_world, tau_total_world
