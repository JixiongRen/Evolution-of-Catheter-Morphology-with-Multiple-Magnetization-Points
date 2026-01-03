
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

from supiee_auto_diff.actuation_interpolator_jax import load_actuation_table, interpolate_A_jax, interpolate_A_vmap


def _as_positions_3xn(pose_list) -> jnp.ndarray:
    if isinstance(pose_list, jnp.ndarray):
        if pose_list.ndim == 1:
            return pose_list.reshape(3, 1)
        return pose_list
    if isinstance(pose_list, np.ndarray):
        if pose_list.ndim == 1:
            return jnp.asarray(pose_list.reshape(3, 1))
        return jnp.asarray(pose_list)

    if not pose_list:
        raise ValueError("pose_list is empty")

    first = pose_list[0]
    if isinstance(first, jnp.ndarray):
        return jnp.stack([jnp.asarray(p).reshape(3,) for p in pose_list], axis=1)

    positions = np.stack([np.asarray(p).reshape(3,) for p in pose_list], axis=1)
    return jnp.asarray(positions)


def _as_moment_3xn(magnetic_moment) -> jnp.ndarray:
    if isinstance(magnetic_moment, jnp.ndarray):
        if magnetic_moment.ndim == 1:
            return magnetic_moment.reshape(3, 1)
        return magnetic_moment
    magnetic_moment = np.asarray(magnetic_moment)
    if magnetic_moment.ndim == 1:
        magnetic_moment = magnetic_moment.reshape(3, 1)
    return jnp.asarray(magnetic_moment)


@jax.jit
def _wrench_from_BG5(B: jnp.ndarray, G5: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    mx, my, mz = m[0], m[1], m[2]

    force_matrix = jnp.array(
        [
            [mx, my, mz, 0.0, 0.0],
            [0.0, mx, 0.0, my, mz],
            [-mz, 0.0, mx, -mz, my],
        ],
        dtype=B.dtype,
    )
    Fm = force_matrix @ G5

    torque_matrix = jnp.array(
        [
            [0.0, -mz, my],
            [mz, 0.0, -mx],
            [-my, mx, 0.0],
        ],
        dtype=B.dtype,
    )
    Tm = torque_matrix @ B

    return jnp.concatenate([Fm, Tm], axis=0)


@jax.jit
def _magnetic_wrench_core(
    A_table: jnp.ndarray,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    zs: jnp.ndarray,
    positions_3xn: jnp.ndarray,
    magnetic_moment_3xn: jnp.ndarray,
    currents_vector: jnp.ndarray,
) -> jnp.ndarray:
    P = positions_3xn.T
    m = magnetic_moment_3xn.T

    A = interpolate_A_vmap(A_table, xs, ys, zs, P)
    y8 = jnp.einsum("nij,j->ni", A, currents_vector)
    B = y8[:, 0:3]
    G5 = y8[:, 3:8]

    wrench_n6 = jax.vmap(_wrench_from_BG5)(B, G5, m)
    return wrench_n6.T


class MagneticActuationSystem:
    def __init__(
        self,
        calib_file: str,
        *,
        actuation_table_pkl: str | None = None,
        dtype=jnp.float32,
        enable_x64: bool = False,
    ):
        if enable_x64:
            jax.config.update("jax_enable_x64", True)

        if actuation_table_pkl is None:
            repo_root = Path(__file__).resolve().parents[1]
            actuation_table_pkl = str(
                (
                    repo_root
                    / "supiee_auto_diff"
                    / "offline_interpolation_data"
                    / "actuation_tables"
                    / "actuation_table.pkl"
                ).resolve()
            )

        self.calib_file = calib_file
        self._table = load_actuation_table(actuation_table_pkl, dtype=dtype)
        self._interp_A = interpolate_A_jax

    def actuation_matrix(self, position: jnp.ndarray) -> jnp.ndarray:
        position = jnp.asarray(position).reshape(3,)
        return self._interp_A(self._table.A_table, self._table.xs, self._table.ys, self._table.zs, position)

    def y8(self, position: jnp.ndarray, currents_vector: jnp.ndarray) -> jnp.ndarray:
        position = jnp.asarray(position).reshape(3,)
        currents_vector = jnp.asarray(currents_vector).reshape(8,)
        A = self.actuation_matrix(position)
        return A @ currents_vector

    def b_field(self, position: jnp.ndarray, currents_vector: jnp.ndarray) -> jnp.ndarray:
        return self.y8(position, currents_vector)[0:3]

    def b_field_gradient(self, position: jnp.ndarray, currents_vector: jnp.ndarray) -> jnp.ndarray:
        return self.y8(position, currents_vector)[3:8]

    def magnetic_wrench(self, pose_list: list, magnetic_moment: jnp.ndarray, currents_vector: jnp.ndarray) -> jnp.ndarray:
        positions_3xn = _as_positions_3xn(pose_list)
        magnetic_moment_3xn = _as_moment_3xn(magnetic_moment)
        currents_vector = jnp.asarray(currents_vector).reshape(8,)
        return _magnetic_wrench_core(
            self._table.A_table,
            self._table.xs,
            self._table.ys,
            self._table.zs,
            positions_3xn,
            magnetic_moment_3xn,
            currents_vector,
        )

from pose_modules_nondim_jax.external_wrench_nondim_jax import MagneticModel


@dataclass
class SupieeMagneticModel(MagneticModel):
    mas: MagneticActuationSystem

    def wrench_on_magnet(
        self,
        position_world: jnp.ndarray,
        R_world_from_body: jnp.ndarray,
        magnet_params: Dict,
        coil_currents: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        m_body = jnp.asarray(magnet_params["m_body"]).reshape(3,)
        m_world = jnp.asarray(R_world_from_body) @ m_body
        pose = jnp.asarray(position_world).reshape(3, 1)
        m_vec = m_world.reshape(3, 1)
        wrench = self.mas.magnetic_wrench(pose_list=pose, magnetic_moment=m_vec, currents_vector=coil_currents)
        F_world = wrench[:3, 0]
        Tau_world = wrench[3:, 0]
        return F_world, Tau_world
