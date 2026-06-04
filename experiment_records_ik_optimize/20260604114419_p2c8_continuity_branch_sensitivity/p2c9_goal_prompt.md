# P2c9 中长期 Goal Prompt

```text
/goal 在长期无人值守窗口内推进 P2c9，目标是获得跨工况稳定、物理可解释、可审计的 shooting IK 灵敏度 dg/dI，并明确失败后纠偏路径。过程中节省 token，长任务低频检查；文档中文，代码注释/docstring 英文；实验文件只写入工作区；每轮结束提交到本地 ik_optimize 分支，不 push。

Context:
- P2c7 证明原始 18D segment-wrench shooting 的 dg/dI 不可信：fixed-y 显式响应近 0，implicit/GN chain 与 re-optimized FD 不一致，stationarity 高。
- P2c8 证明段间 wrench continuity 是关键病因之一：soft continuity 可把 stationarity 压低，并让部分方向灵敏度接近 FD；但 active coil 方向仍失败，结论只能是 partially-usable。
- FK oracle 只作为弱对照，不作为唯一真值；禁止把 oracle centerline、rigid midpoint 或内部形态监督混入主结论。

Success Criteria:
- 至少 2 个温和 case 与 2 个 hard current case 通过审计。
- 每个 case 至少 3 个电流方向，含 active coil 与随机方向。
- terminal position/SO(3)、boundary residual、continuity defect、stationarity/KKT residual 均达标。
- implicit/KKT sensitivity 与 re-optimized FD 的方向导数相对误差 median <=0.3，且无系统性反向；若某些方向物理近零，需通过 eps ladder 证明。
- 明确输出 usable / partially-usable / diagnostic-only / untrusted 分类，不能用少量好行掩盖失败行。

Constraints:
- 不降低原有求解精度门限。
- 不依赖 oracle_y0 宣称 production；如使用 seed/continuation，必须记录 branch policy。
- 不把 sampled Jacobian fallback 当完整 8D production gradient。
- 不追求速度优先；先找对 formulation，再优化运行时间。

Checklist:
1. 完善灵敏度审计门禁：stationarity/KKT 前置门、eps ladder、plus/minus solve stability、conditioning、branch jump、failure taxonomy。
2. 实现 P2c9a：multiple shooting + hard wrench continuity，用等式约束或增广 KKT 替代 soft penalty；输出 KKT residual 与隐式灵敏度。
3. 实现 P2c9b：true continuous initial-value shooting，变量优先为 base wrench 6D；必要时增加少量低阶 mode/slack；用 current/target continuation 和 multi-start 保支。
4. 与 P2c8 cont_1.0 soft penalty 进行同工况对比，只把 soft penalty 作为诊断 fallback。
5. 若 P2c9a 失败：定位是 scaling/conditioning、可观测性不足、hard constraint 不可行，还是 branch jump；然后调整缩放、增广拉格朗日或 pseudo-arclength continuation。
6. 若 P2c9b 失败：定位是自由度不足、吸引域不足还是模型结构不匹配；必要时转向 base wrench + interface slack 重参数化。
7. 形成实验报告、summary JSON/CSV、failure_events、heartbeat、关键图像和下一步结论；本地 commit 到 ik_optimize。

End State:
- 若成功，给出可进入外层 IK 的 dg/dI 接口与适用边界。
- 若未成功，给出最可信 formulation、失败主因、下一轮明确修复方案，不中途停在单个短期验证失败。
```
