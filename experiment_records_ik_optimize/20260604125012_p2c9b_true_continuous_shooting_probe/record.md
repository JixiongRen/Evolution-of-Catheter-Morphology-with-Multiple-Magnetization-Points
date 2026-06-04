# P2c9b 真连续初值打靶原型实验记录

## 实验目的

P2c9 严格门禁显示，soft 段间连续性罚项只能得到部分可用的 `dg/dI`。本轮实验验证另一条 formulation 级路线：真连续初值打靶，即只优化一个基座内力/内矩 `6D` 变量，并让 wrench 沿整根导管连续传播，不再在每个磁体段入口重置。

## 实验方案

变量：

- `y = [n0, m0]`，共 `6D`。

残差：

- terminal position residual。
- terminal SO(3) residual。
- final boundary residual。
- weak base-wrench regularization。

不使用 centerline oracle，不使用刚性段中点监督。FK oracle 只用于生成目标 terminal pose 和初始参考，不作为唯一真值。

## 实验过程记录

运行命令见 `commands.sh`。输出：

- `summary.json`
- `summary.csv`
- `solve_summary.csv`
- `heartbeat.jsonl`
- `failure_events.jsonl`
- `sensitivity_audit.json`

## 实验结果

四个 P2c4 case 的 baseline 全部失败，因此没有进入正式灵敏度审计。

结果摘要：

- 总体分类：`untrusted`。
- `n3_mixed_base_i4`：`residual_norm≈4.03`，`stationarity_norm≈3170.63`，`boundary≈7.05`。
- `n3_mixed_rand20`：`residual_norm≈2.76`，`stationarity_norm≈38632.31`，`boundary≈0.230`。
- `n2_z_pair_rand12`：`residual_norm≈0.313`，`stationarity_norm≈16997.20`，`boundary≈0.587`。
- `n1_tip_z_rand15`：`residual_norm≈0.340`，`stationarity_norm≈8351.52`，`boundary≈0.291`。

各 case 的 steady solve time 约 `1.14-1.30 s`，说明失败不是因为计算时间不足，而是 formulation 自由度或吸引域不足。

## 数据分析

重点判断：

- 6D true continuous shooting 是否能在四个 P2c4 case 上形成低 residual、低 stationarity 的基线。
- 若能形成基线，再检查 implicit chain 与 re-optimized FD 是否一致。
- 若基线失败，说明单纯 6D 初值打靶自由度不足或吸引域不足，需要引入 continuation、low-order slack 或 hard multiple shooting。

本轮数据支持如下判断：

- 纯 6D base-wrench shooting 过于刚性，无法同时匹配 terminal pose 和 final boundary。
- `n3_mixed_base_i4` 的 boundary residual 极大，说明连续传播的单一初值无法表达当前多磁体段下的内力/内矩重分配。
- 与 P2c9 的 soft continuity 结果相比，纯 6D 路线更物理但欠自由度；18D soft penalty 自由度足够但灵敏度受权重污染。
- 因此更合理的中间路线是 `base wrench + interface slack`：保持主链连续，同时允许每个接口存在小的、受正则约束的 wrench jump。

## 实验结论

P2c9b 作为独立 6D true continuous IVP shooting 路线失败，不能作为当前 IK 灵敏度主线。

这不是目标失败，而是纠偏信息：当前问题需要在“物理连续性”和“足够表达能力”之间取中间 formulation。下一步不应继续调 6D 初值打靶，而应实现连续主链 + interface slack，或直接实现 hard-continuity multiple shooting + KKT/IFT。

## 下一步计划

下一步建议：

- P2c9c：实现 `base wrench + interface slack` formulation。变量为 `w0 + δw_i`，默认沿链连续传播，只允许接口有小 slack；对 slack 加物理正则。
- P2c9d：实现 hard-continuity multiple shooting + KKT/IFT，将段间 continuity 作为等式约束而非 soft penalty。
- 优先用 P2c9c 做轻量验证，因为它比完整 KKT 系统工程量小，同时比纯 6D 有更多自由度。
