# nondim.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class NondimScales:
    """
    Global nondimensional scales:
      - L_ref: length scale [m]
      - F_ref: force scale [N]
      - M_ref: moment scale [N*m]
    """
    L_ref: float
    F_ref: float
    M_ref: float

    def __post_init__(self):
        if self.L_ref <= 0 or self.F_ref <= 0 or self.M_ref <= 0:
            raise ValueError("All scales must be positive.")


def compute_default_scales(
        flex_segs: List,
        rigid_segs: Optional[List] = None,
        ref_flex_index: int = 0,
) -> NondimScales:
    """
    Recommend default scales:
      L_ref = total catheter length (sum of all flex + rigid)
      F_ref = mean(diag(K_se)) of a reference flexible segment
      M_ref = mean(diag(K_bt)) / L_ref of the same segment
    """
    if not flex_segs:
        raise ValueError("flex_segs is empty.")

    L_ref = float(sum(seg.length for seg in flex_segs))
    if rigid_segs is not None:
        L_ref += float(sum(seg.length for seg in rigid_segs))

    ref = flex_segs[ref_flex_index]
    Kse = np.asarray(ref.K_se, dtype=float)
    Kbt = np.asarray(ref.K_bt, dtype=float)

    F_ref = float(np.mean(np.diag(Kse)))
    # M_ref = float(np.mean(np.diag(Kbt)) / L_ref)
    M_ref = F_ref * L_ref

    # safety floors
    eps = 1e-12
    F_ref = max(F_ref, eps)
    M_ref = max(M_ref, eps)
    L_ref = max(L_ref, eps)

    return NondimScales(L_ref=L_ref, F_ref=F_ref, M_ref=M_ref)


# ----------------- x scaling (13-dim state) -----------------

def x_dim_to_bar(x_dim: np.ndarray, s: NondimScales) -> np.ndarray:
    """
    x = [p(3), Q(4), f(3), tau(3)] in SI  ->  x_bar nondimensional
      p_bar   = p / L_ref
      Q_bar   = Q (unchanged)
      f_bar   = f / F_ref
      tau_bar = tau / M_ref
    """
    x_dim = np.asarray(x_dim, dtype=float).reshape(13)
    x_bar = x_dim.copy()
    x_bar[:3] = x_dim[:3] / s.L_ref
    # Q unchanged
    x_bar[7:10] = x_dim[7:10] / s.F_ref
    x_bar[10:13] = x_dim[10:13] / s.M_ref
    return x_bar


def x_bar_to_dim(x_bar: np.ndarray, s: NondimScales) -> np.ndarray:
    """
    x_bar nondimensional -> x in SI
    """
    x_bar = np.asarray(x_bar, dtype=float).reshape(13)
    x_dim = x_bar.copy()
    x_dim[:3] = x_bar[:3] * s.L_ref
    # Q unchanged
    x_dim[7:10] = x_bar[7:10] * s.F_ref
    x_dim[10:13] = x_bar[10:13] * s.M_ref
    return x_dim


def rhs_dim_to_rhs_bar_dsbar(dx_dsigma_dim: np.ndarray, L_seg: float, s: NondimScales) -> np.ndarray:
    """
    Convert derivative w.r.t. sigma (meters) into derivative of x_bar w.r.t. s_bar in [0,1]:
        s_bar = sigma / L_seg
        d x_bar / d s_bar = L_seg * d x_bar / d sigma
        d x_bar / d sigma = scale(d x / d sigma)
    """
    dx_dsigma_dim = np.asarray(dx_dsigma_dim, dtype=float).reshape(13)
    out = dx_dsigma_dim.copy()

    # scale d/dsigma to d/dsigma for x_bar
    out[:3] = dx_dsigma_dim[:3] / s.L_ref
    # Q unchanged
    out[7:10] = dx_dsigma_dim[7:10] / s.F_ref
    out[10:13] = dx_dsigma_dim[10:13] / s.M_ref

    # chain rule: multiply by L_seg
    return float(L_seg) * out
