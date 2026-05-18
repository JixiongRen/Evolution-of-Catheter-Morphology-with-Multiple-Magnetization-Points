from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .segments import FlexibleSegment


@dataclass
class RodMesh:
    """Discretize a single flexible segment along arc length."""

    flex_seg: FlexibleSegment
    n_intervals: int = 1
    sigma_nodes: Optional[np.ndarray] = None  # shape (M+1,)

    def __post_init__(self):
        L = self.flex_seg.length
        if self.sigma_nodes is None:
            self.sigma_nodes = np.linspace(0.0, L, self.n_intervals + 1)
        else:
            self.sigma_nodes = np.asarray(self.sigma_nodes).ravel()
            assert self.sigma_nodes[0] == 0.0
            assert self.sigma_nodes[-1] == L
            self.n_intervals = self.sigma_nodes.size - 1

        self.h_list = self.sigma_nodes[1:] - self.sigma_nodes[:-1]

    @property
    def M(self) -> int:
        """Return the number of mesh intervals."""
        return self.n_intervals
