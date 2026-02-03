# nondim.py
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NondimScales:
    L_ref: float
    F_ref: float
    M_ref: float

    def tree_flatten(self):
        children = (
            jnp.asarray(self.L_ref, dtype=jnp.float64),
            jnp.asarray(self.F_ref, dtype=jnp.float64),
            jnp.asarray(self.M_ref, dtype=jnp.float64),
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        L_ref, F_ref, M_ref = children
        return cls(L_ref=L_ref, F_ref=F_ref, M_ref=M_ref)

def x_bar_to_dim(x_bar: jnp.ndarray, s: NondimScales) -> jnp.ndarray:
    x_bar = jnp.asarray(x_bar).reshape(13,)
    x_dim = x_bar
    x_dim = x_dim.at[0:3].set(x_bar[0:3] * s.L_ref)
    x_dim = x_dim.at[7:10].set(x_bar[7:10] * s.F_ref)
    x_dim = x_dim.at[10:13].set(x_bar[10:13] * s.M_ref)
    return x_dim

def x_dim_to_bar(x_dim: jnp.ndarray, s: NondimScales) -> jnp.ndarray:
    x_dim = jnp.asarray(x_dim).reshape(13,)
    x_bar = x_dim
    x_bar = x_bar.at[0:3].set(x_dim[0:3] / s.L_ref)
    x_bar = x_bar.at[7:10].set(x_dim[7:10] / s.F_ref)
    x_bar = x_bar.at[10:13].set(x_dim[10:13] / s.M_ref)
    return x_bar

def rhs_dim_to_rhs_bar_dsbar(dx_dsigma_dim: jnp.ndarray, L_seg: float, s: NondimScales) -> jnp.ndarray:
    dx = jnp.asarray(dx_dsigma_dim).reshape(13,)
    out = dx
    out = out.at[0:3].set(dx[0:3] / s.L_ref)
    out = out.at[7:10].set(dx[7:10] / s.F_ref)
    out = out.at[10:13].set(dx[10:13] / s.M_ref)
    return jnp.asarray(L_seg, dtype=out.dtype) * out
