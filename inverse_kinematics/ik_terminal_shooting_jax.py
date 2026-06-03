"""JAX clean terminal-pose shooting utilities.

This module implements the segment-wrench shooting path used by the P2c clean
terminal-pose experiments. The residual intentionally contains only terminal
position, terminal SO(3), and final physical boundary terms.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
if str(FK_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(FK_OPT_DIR))

from basics_nondim import quat_normalize
from external_wrench_nondim import compute_external_wrench_total_rigid
from forward_kinematics_optimized.segments_nondim import cosserat_rhs_dim, rigid_state_along_dim
from inverse_kinematics.ik_observations import quaternion_so3_residual

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


@dataclass(frozen=True)
class CleanShootingWeights:
    """Weights for the clean terminal-pose shooting residual."""

    tip_weight: float = 120.0
    so3_weight: float = 2.0
    boundary_weight: float = 0.02


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CleanShootingOutput:
    """Output from one segment-wrench shooting integration."""

    centerline: Array
    rigid_mid_p: Array
    rigid_mid_q: Array
    tip_p: Array
    tip_q: Array
    boundary: Array

    def tree_flatten(self):
        children = (self.centerline, self.rigid_mid_p, self.rigid_mid_q, self.tip_p, self.tip_q, self.boundary)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        centerline, rigid_mid_p, rigid_mid_q, tip_p, tip_q, boundary = children
        return cls(
            centerline=centerline,
            rigid_mid_p=rigid_mid_p,
            rigid_mid_q=rigid_mid_q,
            tip_p=tip_p,
            tip_q=tip_q,
            boundary=boundary,
        )


def normalize_state(x: Array) -> Array:
    """Normalize the quaternion block of a 13-state vector."""
    x = jnp.asarray(x, dtype=jnp.float64).reshape(13)
    return x.at[3:7].set(quat_normalize(x[3:7]))


def initial_state(params: Any, first_block: Array) -> Array:
    """Build the dimensional initial shooting state."""
    first_block = jnp.asarray(first_block, dtype=jnp.float64).reshape(6)
    x = jnp.zeros((13,), dtype=jnp.float64)
    x = x.at[0:3].set(jnp.asarray(params.p0_bar, dtype=jnp.float64).reshape(3) * float(params.scales.L_ref))
    x = x.at[3:7].set(jnp.asarray(params.Q0, dtype=jnp.float64).reshape(4))
    x = x.at[7:10].set(first_block[0:3] * float(params.scales.F_ref))
    x = x.at[10:13].set(first_block[3:6] * float(params.scales.M_ref))
    return normalize_state(x)


def rk4_step(rhs: Callable[[Array, Array], Array], x: Array, sigma: Array, h: Array) -> Array:
    """Advance one RK4 step and normalize the quaternion after substeps."""
    k1 = rhs(x, sigma)
    k2 = rhs(normalize_state(x + 0.5 * h * k1), sigma + 0.5 * h)
    k3 = rhs(normalize_state(x + 0.5 * h * k2), sigma + 0.5 * h)
    k4 = rhs(normalize_state(x + h * k3), sigma + h)
    return normalize_state(x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _integrate_flex(params: Any, x0: Array, i: int, steps_per_flex: int) -> tuple[Array, Array]:
    flex = params.flex[i]
    length = jnp.asarray(params.L1_dim if i == 0 else flex.length, dtype=jnp.float64).reshape(())
    n_steps = max(int(steps_per_flex), int(params.M_list[i]) * int(steps_per_flex))
    h = length / float(n_steps)
    f_line = jnp.asarray(params.flex_f_line_world_list[i], dtype=jnp.float64).reshape(3)
    tau_line = jnp.asarray(params.flex_tau_line_world_list[i], dtype=jnp.float64).reshape(3)

    def rhs(x: Array, sigma: Array) -> Array:
        return cosserat_rhs_dim(
            x,
            sigma,
            Kse_inv=flex.Kse_inv,
            Kbt_inv=flex.Kbt_inv,
            v_star=flex.v_star,
            u_star=flex.u_star,
            fext_density=lambda _x, _s: f_line,
            tauext_density=lambda _x, _s: tau_line,
        )

    def body(carry: tuple[Array, Array], _idx: Array) -> tuple[tuple[Array, Array], Array]:
        x, sigma = carry
        x_next = rk4_step(rhs, x, sigma, h)
        return (x_next, sigma + h), x_next[0:3]

    (x_end, _sigma_end), flex_points = jax.lax.scan(body, (x0, jnp.asarray(0.0, dtype=jnp.float64)), jnp.arange(n_steps))
    return x_end, flex_points


def _advance_rigid(params: Any, x_prox: Array, i: int) -> tuple[Array, Array, Array]:
    rigid = params.rigid[i]
    f_total, tau_total = compute_external_wrench_total_rigid(
        x_proximal=x_prox,
        rigid_length=float(rigid.length),
        gravity=params.gravity_rigid_list[i],
        magnetic_model=params.magnetic_model,
        magnet_params=params.magnet_params_list[i],
        coil_currents=params.coil_currents,
    )
    f_total = f_total + jnp.asarray(params.f_ext_list[i], dtype=jnp.float64).reshape(3)
    tau_total = tau_total + jnp.asarray(params.tau_ext_list[i], dtype=jnp.float64).reshape(3)
    x_mid = rigid_state_along_dim(
        x_prox_dim=x_prox,
        sigma=0.5 * float(rigid.length),
        rigid=rigid,
        f_ext_total_dim=f_total,
        tau_ext_total_dim=tau_total,
    )
    x_dist = rigid_state_along_dim(
        x_prox_dim=x_prox,
        sigma=float(rigid.length),
        rigid=rigid,
        f_ext_total_dim=f_total,
        tau_ext_total_dim=tau_total,
    )
    return normalize_state(x_dist), x_mid[0:3], quat_normalize(x_mid[3:7])


def forward_segment_wrench(params: Any, y: Array, *, steps_per_flex: int = 4) -> CleanShootingOutput:
    """Integrate the 18D segment-wrench shooting state."""
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = initial_state(params, blocks[0])
    centerline_parts = [x[0:3].reshape(1, 3)]
    rigid_mid_p = []
    rigid_mid_q = []

    for i in range(len(params.flex)):
        x = x.at[7:10].set(blocks[i, 0:3] * float(params.scales.F_ref))
        x = x.at[10:13].set(blocks[i, 3:6] * float(params.scales.M_ref))
        x = normalize_state(x)
        x, flex_points = _integrate_flex(params, x, i, steps_per_flex)
        centerline_parts.append(flex_points)
        x, mid_p, mid_q = _advance_rigid(params, x, i)
        rigid_mid_p.append(mid_p)
        rigid_mid_q.append(mid_q)
        centerline_parts.append(x[0:3].reshape(1, 3))

    boundary = jnp.concatenate(
        [
            x[7:10] / jnp.maximum(float(params.scales.F_ref), 1e-12),
            x[10:13] / jnp.maximum(float(params.scales.M_ref), 1e-12),
        ],
        axis=0,
    )
    return CleanShootingOutput(
        centerline=jnp.concatenate(centerline_parts, axis=0),
        rigid_mid_p=jnp.stack(rigid_mid_p, axis=0),
        rigid_mid_q=jnp.stack(rigid_mid_q, axis=0),
        tip_p=x[0:3],
        tip_q=quat_normalize(x[3:7]),
        boundary=boundary,
    )


def clean_terminal_residual(
    params: Any,
    y: Array,
    *,
    target_p: Array,
    target_q: Array,
    weights: CleanShootingWeights = CleanShootingWeights(),
    steps_per_flex: int = 4,
    so3_scale: float = 1.0,
) -> Array:
    """Return the clean residual for terminal-pose shooting."""
    out = forward_segment_wrench(params, y, steps_per_flex=steps_per_flex)
    r_p = float(weights.tip_weight) * (out.tip_p - jnp.asarray(target_p, dtype=jnp.float64).reshape(3))
    r_q = float(weights.so3_weight) * jnp.asarray(so3_scale, dtype=jnp.float64) * quaternion_so3_residual(
        out.tip_q,
        jnp.asarray(target_q, dtype=jnp.float64).reshape(4),
        sigma_angle=1.0,
    )
    r_b = float(weights.boundary_weight) * out.boundary
    return jnp.concatenate([r_p.reshape(-1), r_q.reshape(-1), r_b.reshape(-1)], axis=0)


def output_to_numpy(out: CleanShootingOutput) -> dict[str, Any]:
    """Convert a JAX shooting output to NumPy arrays."""
    return {
        "centerline": jax.device_get(out.centerline),
        "rigid_mid_p": jax.device_get(out.rigid_mid_p),
        "rigid_mid_q": jax.device_get(out.rigid_mid_q),
        "tip_p": jax.device_get(out.tip_p),
        "tip_q": jax.device_get(out.tip_q),
        "boundary": jax.device_get(out.boundary),
    }
