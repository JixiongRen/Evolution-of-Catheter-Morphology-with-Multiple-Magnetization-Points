# P2c139 单点中段磁体分支恢复小实验

## 实验目的

P2c132 表明 `n1_mid_z_rand15_a010` 只有 FK oracle 初值通过，其他合理初值因条件数病态失败。本轮尝试电流与目标同步递增的 continuation 路径，检验是否能获得第二条可信路径。

## 实验方案

固定中段单磁体轴向磁化和最终电流比例 `alpha=0.10`。测试三类路径：直接 oracle 初值、直接 zero_state 初值、从零电流状态开始的两条电流-目标 continuation 路径。每个中间 alpha 都重新生成对应 FK oracle 目标，并使用 P2c132 相同的显式刚段载荷残差、三维末端位置任务和门禁。

## 结果

- direct_oracle：基线门禁 True；位置误差 3.296e-07 m；一阶最优性 7.242e-11；条件数 8.243e+05；失败分类 通过。
- direct_zero_state：基线门禁 False；位置误差 2.636e-06 m；一阶最优性 5.087e-08；条件数 1.535e+07；失败分类 条件数病态。
- cont_coarse：基线门禁 True；位置误差 2.473e-06 m；一阶最优性 9.540e-10；条件数 3.685e+06；失败分类 通过。
- cont_fine：基线门禁 True；位置误差 2.473e-06 m；一阶最优性 3.167e-10；条件数 3.685e+06；失败分类 通过。

## 分支漂移

- direct_oracle vs direct_zero_state：两者均过基线 False；最大中心线漂移 7.411e-03 m；分支门禁 False。
- direct_oracle vs cont_coarse：两者均过基线 True；最大中心线漂移 6.304e-03 m；分支门禁 False。
- direct_oracle vs cont_fine：两者均过基线 True；最大中心线漂移 6.304e-03 m；分支门禁 False。
- direct_zero_state vs cont_coarse：两者均过基线 False；最大中心线漂移 7.689e-03 m；分支门禁 False。
- direct_zero_state vs cont_fine：两者均过基线 False；最大中心线漂移 7.689e-03 m；分支门禁 False。
- cont_coarse vs cont_fine：两者均过基线 True；最大中心线漂移 3.132e-17 m；分支门禁 True。

## 结论

分支恢复结论：通过。本轮恢复出一个由 `cont_coarse` 与 `cont_fine` 共同支持的 continuation 分支，两条路径均过基线门禁且中心线漂移接近零；但该分支与 direct_oracle 分支中心线最大漂移约 `6.3 mm`，说明系统存在多平衡分支竞争。

因此，n1 中段候选不能发布为无条件全局灵敏度；但可以进入“continuation 分支条件下”的小扰动重求解复核。后续扰动必须从 continuation 分支出发，并持续检查是否离开该分支。

## 下一步计划

下一步建议执行 P2c140：只在 continuation 分支上做小电流扰动重求解和局部线性化检查。若扰动后仍保持分支一致、线性误差小于 `0.05`、载荷传播受控，则 n1 可形成“分支条件下的有效子空间候选”；否则维持不可发布结论。

## 耗时

- 墙钟耗时：3046.01 s。
- 重求解净耗时：1275.22 s。
