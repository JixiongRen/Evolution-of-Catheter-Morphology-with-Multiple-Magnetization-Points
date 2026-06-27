# P2c76 n2 完整最优性灵敏度记录

## 实验目的

P2c75 表明误差来自状态灵敏度线性方程。本轮使用最小二乘一阶最优性 `g(q,I)=J(q,I)^T r(q,I)` 的完整导数，检验是否能恢复与扰动重求解一致的局部灵敏度。

## 基线

- 末端位置误差：`1.402e-06 m`
- 一阶最优性：`2.232e-07`
- 条件数：`4.905e+07`
- 接口连续：`4.371e-04`
- 载荷传播比：`7.409e-01`

## 结果

| 扰动尺度 | 方向 | 旧状态误差 | 完整状态误差 | 旧末端误差 | 完整末端误差 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0025 | max_active_coil | 3.165e-02 | 3.195e-03 | 1.952e-01 | 5.294e-04 | linearization_pass_condition_untrusted |
| 0.0025 | current_direction | 9.882e-02 | 4.043e-03 | 1.863e-01 | 4.022e-04 | linearization_pass_condition_untrusted |
| 0.0025 | random_unit | 5.069e-02 | 3.748e-03 | 1.050e-01 | 5.759e-04 | linearization_pass_condition_untrusted |
| 0.005 | max_active_coil | 2.754e-02 | 9.927e-04 | 1.960e-01 | 2.041e-04 | linearization_pass_condition_untrusted |
| 0.005 | current_direction | 9.882e-02 | 4.043e-03 | 1.863e-01 | 4.023e-04 | linearization_pass_condition_untrusted |
| 0.005 | random_unit | 4.589e-02 | 1.167e-03 | 1.058e-01 | 2.690e-04 | linearization_pass_condition_untrusted |
| 0.01 | max_active_coil | 2.881e-02 | 2.997e-04 | 1.957e-01 | 5.034e-05 | linearization_pass_condition_untrusted |
| 0.01 | current_direction | 9.884e-02 | 1.240e-03 | 1.869e-01 | 8.312e-05 | linearization_pass_condition_untrusted |
| 0.01 | random_unit | 4.589e-02 | 1.167e-03 | 1.058e-01 | 2.690e-04 | linearization_pass_condition_untrusted |

## 结论

本轮共有 `0` / `9` 个方向-尺度组合达到完整发布门禁。完整最优性公式将末端误差降到发布阈值以内，但最优性线性系统条件数约为 `2.432e+13`，违反条件数硬门禁。因此当前结论是“线性化机制被修复”，不是“可发布灵敏度已获得”。

- 总耗时：`525.60 s`

## 下一步计划

1. 优先做最优性线性系统的秩截断、阻尼和尺度化审计，降低条件数或给出不可发布边界。
2. 在条件数门禁通过前，不扩展大规模扰动扫描，也不接入外层 IK。
3. 若稳定化后仍保持末端误差低于发布门禁，再做多初值分支复核。
