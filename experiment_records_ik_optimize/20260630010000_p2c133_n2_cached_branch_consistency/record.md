# P2c133 两点磁控缓存分支一致性复核

## 实验目的

P2c125 已将两点磁控定位为“同一可信分支下的局部位置诊断”。本轮检查 P2c104 缓存可信分支在有效子空间内是否对不同初值稳定收敛，为后续三维位置灵敏度发布边界提供分支一致性证据。

## 实验方案

固定 P2c104 的两点磁控可信缓存分支、同一目标点和同一基准电流。在有效子空间中设置零初值、小随机正负扰动、中等随机扰动、第一基向量正负扰动，共六个初值。每个初值重求解同一三维末端位置任务，并比较最终中心线。

通过标准：每条路径需通过位置误差、一阶最优性、有效条件数、接口连续残差和载荷传播门禁；至少一对均过基线路径的中心线最大漂移不超过 `1e-3 m`。

## 基线结果

- zero_q：基线门禁 True；位置误差 3.847e-13 m；一阶最优性 9.144e-09；条件数 5.428e+06；失败分类 通过。
- small_random_pos：基线门禁 False；位置误差 3.950e-02 m；一阶最优性 7.979e+01；条件数 2.774e+07；失败分类 任务误差超限。
- small_random_neg：基线门禁 False；位置误差 4.507e-02 m；一阶最优性 1.886e+05；条件数 3.428e+08；失败分类 任务误差超限。
- medium_random：基线门禁 False；位置误差 8.418e-02 m；一阶最优性 1.372e+03；条件数 6.667e+06；失败分类 任务误差超限。
- basis_axis_pos：基线门禁 False；位置误差 5.864e-02 m；一阶最优性 1.771e+04；条件数 9.340e+07；失败分类 任务误差超限。
- basis_axis_neg：基线门禁 False；位置误差 3.178e-02 m；一阶最优性 4.768e+02；条件数 3.007e+09；失败分类 任务误差超限。

## 分支漂移

- zero_q vs small_random_pos：两者均过基线 False；最大中心线漂移 4.029e-02 m；分支门禁 False。
- zero_q vs small_random_neg：两者均过基线 False；最大中心线漂移 4.507e-02 m；分支门禁 False。
- zero_q vs medium_random：两者均过基线 False；最大中心线漂移 8.418e-02 m；分支门禁 False。
- zero_q vs basis_axis_pos：两者均过基线 False；最大中心线漂移 5.864e-02 m；分支门禁 False。
- zero_q vs basis_axis_neg：两者均过基线 False；最大中心线漂移 3.247e-02 m；分支门禁 False。
- small_random_pos vs small_random_neg：两者均过基线 False；最大中心线漂移 1.844e-02 m；分支门禁 False。
- small_random_pos vs medium_random：两者均过基线 False；最大中心线漂移 4.708e-02 m；分支门禁 False。
- small_random_pos vs basis_axis_pos：两者均过基线 False；最大中心线漂移 3.578e-02 m；分支门禁 False。
- small_random_pos vs basis_axis_neg：两者均过基线 False；最大中心线漂移 2.359e-02 m；分支门禁 False。
- small_random_neg vs medium_random：两者均过基线 False；最大中心线漂移 4.218e-02 m；分支门禁 False。
- small_random_neg vs basis_axis_pos：两者均过基线 False；最大中心线漂移 2.884e-02 m；分支门禁 False。
- small_random_neg vs basis_axis_neg：两者均过基线 False；最大中心线漂移 1.536e-02 m；分支门禁 False。
- medium_random vs basis_axis_pos：两者均过基线 False；最大中心线漂移 2.903e-02 m；分支门禁 False。
- medium_random vs basis_axis_neg：两者均过基线 False；最大中心线漂移 5.252e-02 m；分支门禁 False。
- basis_axis_pos vs basis_axis_neg：两者均过基线 False；最大中心线漂移 3.425e-02 m；分支门禁 False。

## 结论

分支一致性结论：未通过。若通过，则 n2 可继续进入扰动重求解和局部线性化复核；若不通过，则维持“方向诊断可用、三维发布不足”的结论。

## 下一步计划

若本轮通过，执行 P2c134：围绕该缓存分支做三维随机方向小扰动重求解复核，并报告局部线性化误差分布。若未通过，继续维持 P2c125 的方向性工程接口，不升级三维位置发布。

## 耗时

- 墙钟耗时：115.85 s。
- 重求解净耗时：38.38 s。
