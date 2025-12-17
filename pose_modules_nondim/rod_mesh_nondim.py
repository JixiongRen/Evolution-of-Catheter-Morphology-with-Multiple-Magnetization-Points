# rod_mesh_nondim.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .segments_nondim import FlexibleSegment


@dataclass
class RodMesh:
    flex_seg: FlexibleSegment
    n_intervals: int = 1
    sigma_nodes: Optional[np.ndarray] = None  # (M+1,)

    def __post_init__(self):
        L = float(self.flex_seg.length)
        if self.sigma_nodes is None:
            self.sigma_nodes = np.linspace(0.0, L, self.n_intervals + 1)
        else:
            self.sigma_nodes = np.asarray(self.sigma_nodes, dtype=float).ravel()
            assert self.sigma_nodes[0] == 0.0
            assert self.sigma_nodes[-1] == L
            self.n_intervals = self.sigma_nodes.size - 1

        self.h_list = self.sigma_nodes[1:] - self.sigma_nodes[:-1]

        # nondim arc-length within this segment
        self.sbar_nodes = self.sigma_nodes / L                     # (M+1,)
        self.hbar_list = self.h_list / L                           # (M,)

    @property
    def M(self) -> int:
        return self.n_intervals
