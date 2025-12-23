# -*- coding: utf-8 -*-
"""
在此脚本实现了下面功能:

已知Supiee系统由8个电磁体组成, 现在需要给创建8组电流
I1 = np.array([1., 0., 0., 0., 0., 0., 0., 0.,], dtype=np.float64)
I2 = np.array([0., 1., 0., 0., 0., 0., 0., 0.,], dtype=np.float64)
I3 = np.array([0., 0., 1., 0., 0., 0., 0., 0.,], dtype=np.float64)
I4 = np.array([0., 0., 0., 1., 0., 0., 0., 0.,], dtype=np.float64)
I5 = np.array([0., 0., 0., 0., 1., 0., 0., 0.,], dtype=np.float64)
I6 = np.array([0., 0., 0., 0., 0., 1., 0., 0.,], dtype=np.float64)
I7 = np.array([0., 0., 0., 0., 0., 0., 1., 0.,], dtype=np.float64)
I8 = np.array([0., 0., 0., 0., 0., 0., 0., 1.,], dtype=np.float64)

并创建采样格栅:
以原点为中心, 创建一个边长为0.2m的立方体, 立方体每条边(含端点)上有10个采样点

分别采集8组电流下在这个立方体上的采样点的磁场, 创建一个张量
shape = [10, 10, 10, 8, 3]

各个维度表示:
10个采样点在x轴上的坐标
10个采样点在y轴上的坐标
10个采样点在z轴上的坐标
8个电磁体
3个磁场分量

将其保存成 pkl, 供后续建模使用
"""

from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np

# 允许两种导入路径，按你项目结构自动适配
from mag_manip import mag_manip
from mag_manip.mag_manip import ForwardModelMPEM


def generate_unit_currents(n_coils: int = 8, dtype=np.float64) -> np.ndarray:
    """返回 (n_coils, n_coils) 的单位电流矩阵，每行是一个单位电流向量。"""
    return np.eye(n_coils, dtype=dtype)


def generate_grid(side: float = 0.2, num: int = 10, dtype=np.float64):
    """
    以原点为中心生成立方体网格:
    返回 grid_xyz: (num, num, num, 3), 以及 xs, ys, zs 轴坐标
    """
    half = side / 2.0
    xs = np.linspace(-half, half, num=num, dtype=dtype)
    ys = np.linspace(-half, half, num=num, dtype=dtype)
    zs = np.linspace(-half, half, num=num, dtype=dtype)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_xyz = np.stack([X, Y, Z], axis=-1)
    return grid_xyz, xs, ys, zs


def compute_batch_B(supiee, positions: np.ndarray, currents: np.ndarray) -> np.ndarray:
    """
    positions: (N,3)
    currents:  (8,)
    return:    (N,3)
    """
    # 尝试：是否支持批量位置 (N,3)
    try:
        B = supiee.computeFieldFromCurrents(position=positions, currents=currents)
        B = np.asarray(B, dtype=np.float64)
        # 期望返 shape (N,3) 或可压平到此形状
        if B.ndim == 2 and B.shape[1] == 3:
            return B
    except Exception:
        print("not support batch position")
        pass

    return np.stack(
        [
            np.asarray(
                supiee.computeFieldFromCurrents(position=p, currents=currents),
                dtype=np.float64,
            ).reshape(3,)
            for p in positions
        ],
        axis=0,
    )


def main():
    # 初始化 Supiee 前向模型并加载标定文件
    supiee = ForwardModelMPEM()
    calib_path = (Path(__file__).resolve().parent / "../calib/mpem_calibration_file_sp=40_order=1.yaml").resolve()
    supiee.setCalibrationFile(str(calib_path))

    # 如下调用即可获得在指定位置和电流组合下的磁场
    # b_field = supiee.computeFieldFromCurrents(position=position, currents=coils_currents).flatten()

    # 单位电流组
    currents = generate_unit_currents(n_coils=8, dtype=np.float64)  # (8,8)

    # 生成网格 (10,10,10,3) 以及坐标轴
    grid_xyz, xs, ys, zs = generate_grid(side=0.6, num=10, dtype=np.float64)
    nx, ny, nz, _ = grid_xyz.shape
    C = currents.shape[0]  # 8

    # 结果张量: [10, 10, 10, 8, 3]
    B_tensor = np.empty((nx, ny, nz, C, 3), dtype=np.float64)

    # 逐点逐电流计算 B 场
    # 逐电流批量计算 B 场（展平 -> 批量计算 -> 还原网格）
    pts_flat = grid_xyz.reshape(-1, 3)  # (N,3)
    for c in range(C):
        coils_currents = currents[c]  # (8,)
        B_flat = compute_batch_B(supiee, pts_flat, coils_currents)  # (N,3)
        B_tensor[:, :, :, c, :] = B_flat.reshape(nx, ny, nz, 3)

    # 保存为 pkl
    out_dir = (Path(__file__).resolve().parent / "offline_interpolation_data/unit_current_b_data").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / "unit_current_impact.pkl").resolve()
    payload = {
        "B": B_tensor,               # (10,10,10,8,3)
        "x": xs, "y": ys, "z": zs,   # 各轴坐标
        "currents": currents,        # (8,8) 单位电流组
        "meta": {
            "units": {"length": "m", "current": "A", "B": "T"},
            "side": 0.6,
            "num_per_edge": 10,
            "shape": list(B_tensor.shape),
            "description": "Unit-current magnetic field impact tensor over a cubic grid centered at origin.",
        }
    }

    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"已保存到: {out_path}")
    print(f"B 张量形状: {B_tensor.shape} (应为 [10, 10, 10, 8, 3])")


if __name__ == "__main__":
    main()

    # out_path = (Path(__file__).resolve().parent / "unit_current_b_data/unit_current_impact.pkl").resolve()
    # # 测试加载
    # with open(out_path, "rb") as f:
    #     payload = pickle.load(f)
    # B = payload["B"]
    #
    # assert B.shape == (10, 10, 10, 8, 3)