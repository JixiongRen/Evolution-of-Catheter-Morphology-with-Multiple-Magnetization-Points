from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp

from .basics_nondim_jax import quat_normalize, quat_to_rotmat

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


# --------------------------- Flexible: distributed loads ---------------------------

@dataclass(frozen=True)
class GravityLineDensity:
    """Flexible-segment gravity as a *line density* in world frame.

    Convention
    ----------
    - force_world() returns a constant vector in SI units [N/m], in WORLD frame.
      i.e., it already includes rho*A*g (not just unit direction).
    """
    f_line_world: Array  # (3,) [N/m] in world frame

    def force_world(self) -> Array:
        return jnp.asarray(self.f_line_world, dtype=jnp.float64).reshape(3,)


class MagneticDensityModel(Protocol):
    """Optional: distributed magnetic force/torque density along flexible segment."""
    def force_torque_density_body(
        self,
        p_world: Array,      # (3,) SI
        R_world_from_body: Array,  # (3,3)
        sigma: Array,        # scalar SI
        **kwargs
    ) -> Tuple[Array, Array]:
        """Return (f_body, tau_body) as densities in BODY frame: [N/m], [(N*m)/m]."""
        ...


def make_external_wrench_density_flexible_jax(
    *,
    gravity: Optional[GravityLineDensity] = None,
    magnetic_density_model: Optional[MagneticDensityModel] = None,
    magnetic_kwargs: Optional[Dict] = None,
) -> Tuple[
    Callable[[Array, Array], Array],
    Callable[[Array, Array], Array],
]:
    """Create JAX-traceable (AD/JIT friendly) distributed load callbacks.

    Returns
    -------
    fext_density(x_dim, sigma) -> (3,) world force density [N/m]
    tauext_density(x_dim, sigma) -> (3,) world torque density [(N*m)/m]
    where x_dim is the 13-state in SI (p,Q,f,tau) but only p,Q are used here.
    """
    if magnetic_kwargs is None:
        magnetic_kwargs = {}

    g_line = None if gravity is None else gravity.force_world()

    def fext_density(x_dim: Array, sigma: Array) -> Array:
        p = x_dim[0:3]
        Q = quat_normalize(x_dim[3:7])
        R = quat_to_rotmat(Q)

        f_world = jnp.zeros((3,), dtype=jnp.float64)
        if g_line is not None:
            f_world = f_world + g_line

        if magnetic_density_model is not None:
            f_body, _ = magnetic_density_model.force_torque_density_body(
                p_world=p, R_world_from_body=R, sigma=sigma, **magnetic_kwargs
            )
            f_world = f_world + (R @ f_body)

        return f_world

    def tauext_density(x_dim: Array, sigma: Array) -> Array:
        p = x_dim[0:3]
        Q = quat_normalize(x_dim[3:7])
        R = quat_to_rotmat(Q)

        tau_world = jnp.zeros((3,), dtype=jnp.float64)

        if magnetic_density_model is not None:
            _, tau_body = magnetic_density_model.force_torque_density_body(
                p_world=p, R_world_from_body=R, sigma=sigma, **magnetic_kwargs
            )
            tau_world = tau_world + (R @ tau_body)

        # gravity torque density is 0 by default (line load through centerline)
        return tau_world

    return fext_density, tauext_density


# ------------------------------ Rigid: lumped loads ------------------------------

@dataclass(frozen=True)
class GravityRigid:
    """Rigid-segment gravity (lumped) about proximal frame.

    Inputs are SI.
    """
    mass: float
    g_world: Array        # (3,)
    r_cm_body: Array      # (3,) COM offset in body frame (from proximal)

    def force_world(self) -> Array:
        return self.mass * jnp.asarray(self.g_world, dtype=jnp.float64).reshape(3,)

    def torque_body(self, Q: Array) -> Array:
        # tau_body = r_cm_body x (R^T f_world)
        R = quat_to_rotmat(quat_normalize(Q))
        f_body = R.T @ self.force_world()
        return jnp.cross(jnp.asarray(self.r_cm_body, dtype=jnp.float64).reshape(3,), f_body)


class MagneticModel(Protocol):
    """Rigid magnetic model (total wrench)."""
    def compute_wrench_world(
        self,
        p_world: Array,  # (3,) SI
        R_world_from_body: Array,  # (3,3)
        magnet_params: Dict,
        coil_currents: Array,
    ) -> Tuple[Array, Array]:
        """Return (f_world, tau_world) as TOTAL wrench on rigid segment."""
        ...


def compute_external_wrench_total_rigid(
    *,
    x_proximal: Array,               # (13,) SI state at rigid proximal
    rigid_length: float,
    gravity: Optional[GravityRigid],
    magnetic_model: Optional[MagneticModel],
    magnet_params: Optional[Dict],
    coil_currents: Optional[Array],
) -> Tuple[Array, Array]:
    """Compute TOTAL external wrench (world) acting on a rigid segment.
    Convention: returned (f_world, tau_world) is about the rigid proximal point, in WORLD frame.
    """
    p = jnp.asarray(x_proximal[0:3], dtype=jnp.float64).reshape(3,)
    Q = quat_normalize(jnp.asarray(x_proximal[3:7], dtype=jnp.float64).reshape(4,))
    R = quat_to_rotmat(Q)

    f_world = jnp.zeros((3,), dtype=jnp.float64)
    tau_world = jnp.zeros((3,), dtype=jnp.float64)

    # gravity
    if gravity is not None:
        f_g = gravity.force_world()
        tau_g_body = gravity.torque_body(Q)  # body
        tau_g_world = R @ tau_g_body
        f_world = f_world + f_g
        tau_world = tau_world + tau_g_world

    # magnetics
    if (magnetic_model is not None) and (magnet_params is not None) and (coil_currents is not None):
        f_m, tau_m = magnetic_model.compute_wrench_world(
            p_world=p,
            R_world_from_body=R,
            magnet_params=magnet_params,
            coil_currents=coil_currents,
        )
        f_world = f_world + f_m
        tau_world = tau_world + tau_m

    return f_world, tau_world
