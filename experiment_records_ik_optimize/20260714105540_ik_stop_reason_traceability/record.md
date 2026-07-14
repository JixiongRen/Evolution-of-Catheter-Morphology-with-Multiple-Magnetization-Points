# IK 外层停止原因可追溯性修正记录

## 一、实验目的

本轮任务服务于 A1/A2 有监督逆运动学长期目标中的“证据闭环”和“外层优化器稳定化”。当前 A1-pt13 存在接近 `5 mm` 阈值但停止原因不透明的问题，原正式实验脚本只能根据最终误差推断 `reached_10mm_only` 等报告层标签，无法区分核心求解器究竟是因为有效步长过小、预测下降过小、连续拒绝或达到迭代上限而停止。

本轮修正的目标是先补齐停止原因证据链，不改变物理模型、FK 容差、网格、真实末端定义、成功阈值、控制边界或接受判据。

## 二、修改内容

1. 在 `inverse_kinematics/ik_multisite_lm_opt.py` 中新增 `outer_stop_reason` 和 `outer_stop_detail`，写入 `result.json`。
2. 对以下核心停止条件写入明确原因：
   - 初始状态已达标；
   - 控制通道全部被门禁阻塞；
   - 有效电流步长和推进步长均低于数值阈值；
   - 预测下降量低于数值阈值；
   - 真实末端达到阈值；
   - 达到最大外层迭代次数。
3. 对有效步长过小和预测下降过小两类直接停止，额外向 `history.json` 写入 `STOP` 行，保存任务秩、奇异值、阻尼、步长、当前误差和灵敏度摘要。
4. 在 `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py` 中保留原报告层 `stop_reason`，同时新增：
   - `report_stop_reason`：原推断式报告分类；
   - `ik_stop_reason`：核心 IK 写出的真实停止原因；
   - `ik_stop_detail`：核心停止原因的数值细节。
5. 将新增字段加入正式汇总 CSV 的固定列，避免批量分析时丢失。

## 三、验证过程

### 3.1 静态验证

执行：

```bash
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  inverse_kinematics/ik_multisite_lm_opt.py \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py
```

结果：通过。

执行：

```bash
git diff --check -- \
  inverse_kinematics/ik_multisite_lm_opt.py \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py
```

结果：通过。Git 仅提示换行符将按仓库配置转换。

### 3.2 GPU 烟测

执行了一组 A1-pt01、零外层迭代的烟测，仅用于验证字段落盘，不作为算法成功率或性能证据。

命令见 `commands.sh`。

输出目录：

```text
DELIVER_TO_EXP/formal_experiments/20260714105253_A1_A2_5mm_supervised_IK_no_gravity/
```

关键结果：

| 字段 | 值 |
|---|---|
| `result.outer_stop_reason` | `max_outer_iter` |
| `result.outer_stop_detail.outer_max_iter` | `0` |
| `formal_point_summary.report_stop_reason` | `max_outer_iter` |
| `formal_point_summary.ik_stop_reason` | `max_outer_iter` |
| 真实末端误差 | `108.852 mm` |
| 墙钟时间 | `69.83 s` |
| 设备 | `NVIDIA GeForce RTX 5060 Laptop GPU` |

由于外层迭代次数为 `0`，误差大是预期现象。该烟测只证明核心停止原因可以写入结果文件，并被正式实验脚本透传到单点摘要。

## 四、结论

本轮已完成外层停止原因的最小可追溯修正。后续 A1-pt13 近阈值停滞诊断可以直接区分报告层“10 mm 内但未达标”和核心求解器真实停止机制，避免继续把 `reached_10mm_only` 误解为算法终止条件。

## 五、下一步

1. 使用新字段复测 A1-pt13 小任务，确认其真实停止原因是有效步长过小、预测下降过小、连续拒绝还是最大迭代。
2. 若确认为步长或预测下降问题，再引入近阈值补救策略，例如重新线性化、小信赖域搜索或经验微搜。
3. 同步回归 A1-pt01，确保任何补救策略不破坏已知成功点。

