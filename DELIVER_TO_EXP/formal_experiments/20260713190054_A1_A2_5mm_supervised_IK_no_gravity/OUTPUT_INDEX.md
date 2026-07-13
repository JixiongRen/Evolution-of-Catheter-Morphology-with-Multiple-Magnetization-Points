# 输出索引

本目录为 A1/A2 单点磁化导管 5.0 mm 有监督 IK 实验输出。

## 核心文件

- `batch_metadata.json`：实验版本、目标圆、阈值、配置文件与标定文件哈希。
- `targets.json`：20 个圆周目标点坐标。
- `summary.json`：完整机器可读汇总结果。
- `summary.csv`：全部运行点的表格汇总。
- `report.md`：本目录内的中文实验报告副本。
- `commands.sh`：复现实验的启动命令。

## 总览图

- `A1_target_circle_true_tip_overview.png`：A1 真实末端落点与目标圆总览。
- `A2_target_circle_true_tip_overview.png`：A2 真实末端落点与目标圆总览。

## 分点输出

每个点目录均包含 `result.json`、`history.json`、`runtime_profile.json`、`formal_point_summary.json`、`diagnostics.png`、`posture_evolution.png` 和 `poses/` 中心线数据。

## 结论索引

- A1：决策 `formal_20pt_forced`，运行点数 3，5.0 mm 成功 1，最大误差 72.626 mm，平均误差 42.659 mm。