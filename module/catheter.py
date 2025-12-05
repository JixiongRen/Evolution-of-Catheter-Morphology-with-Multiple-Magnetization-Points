from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from basic_utils import quat_to_rotmat

from segments import FlexibleSegment

@dataclass
class RodMesh:
    """
    只针对当前这一段 FlexibleSegment 的离散网格管理：
    - 把 [0, L_flex] 划分为 n_intervals 个小区间
    - 或者传入自定义的 sigma_nodes
    """

    flex_seg: FlexibleSegment
    n_intervals: int = 1
    sigma_nodes: Optional[np.ndarray] = None  # shape (M+1,)

    def __post_init__(self):
        L = self.flex_seg.length
        if self.sigma_nodes is None:
            # 均匀划分
            self.sigma_nodes = np.linspace(0.0, L, self.n_intervals + 1)
        else:
            self.sigma_nodes = np.asarray(self.sigma_nodes).ravel()
            assert self.sigma_nodes[0] == 0.0
            assert self.sigma_nodes[-1] == L
            self.n_intervals = self.sigma_nodes.size - 1

        # 每个小区间长度 h_n
        self.h_list = self.sigma_nodes[1:] - self.sigma_nodes[:-1]

    @property
    def M(self) -> int:
        """柔性段区间数量"""
        return self.n_intervals