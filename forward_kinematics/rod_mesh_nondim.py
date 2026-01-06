import jax.numpy as jnp
from dataclasses import dataclass

Array = jnp.ndarray

@dataclass(frozen=True)
class UniformMesh:
    M: int
    sigma_nodes: Array   # (M+1,) SI
    sbar_nodes: Array    # (M+1,) nondim
    hbar_list: Array     # (M,) nondim step lengths


def build_uniform_mesh(length_dim: float, M: int) -> UniformMesh:
    """
    Uniform mesh on [0, L] with M intervals.
    This is sufficient for equilibrium_solver_nondim_jax.SolverParams:
      - sigma_nodes for initial guess
      - sbar_nodes and hbar_list for residual assembly
    """
    L = float(length_dim)
    M = int(M)
    sigma_nodes = jnp.linspace(0.0, L, M + 1, dtype=jnp.float64)
    sbar_nodes = sigma_nodes / L
    hbar_list = sbar_nodes[1:] - sbar_nodes[:-1]
    return UniformMesh(M=M, sigma_nodes=sigma_nodes, sbar_nodes=sbar_nodes, hbar_list=hbar_list)