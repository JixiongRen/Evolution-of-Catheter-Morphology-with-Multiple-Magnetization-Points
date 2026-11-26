# A1/A2 有监督逆运动学 D4 可达性采样报告

## 一、目的

本报告补充独立控制空间采样和局部加密采样，用于判断失败目标是优化器未找到、边界可达，还是当前采样仍未覆盖。该采样不能证明严格不可达。

## 二、边界条件

- 源实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40`
- 输出目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/reachability_sampling_smoke/20260713122811_A1_A2_reachability_sampling`
- git 版本：`3ee47b16157a26273572f1c14f74bc06ba401b3f`
- JAX 设备：`{'devices': ['cuda:0'], 'platforms': ['gpu'], 'has_gpu': True}`
- 基线保持无重力、`M_list=5`、FK 容差 `1e-5`、电流范围 `[-20,20] A`、推进范围 `[0.003001,0.203] m`。

## 三、采样规模

- 新增样本数：2。
- global_halton：2 个。

## 四、D4 分类结论

### A1
- 边界可达或优化不足：6 点。
- 采样或历史证据可达：7 点。
- 采样或历史证据近可达：3 点。
- 采样仍未覆盖：4 点。

### A2
- 边界可达或优化不足：8 点。
- 采样仍未覆盖：7 点。
- 采样或历史证据可达：4 点。
- 采样或历史证据近可达：1 点。

## 五、解释

1. 若目标被新增采样或历史轨迹覆盖到 5 mm 内，归为已有可达证据；若进入 10 mm，归为近可达；若进入 30 mm，归为边界可达或优化不足。
2. 采样仍未覆盖的点不能写成物理不可达，只能说明当前样本规模不足或目标处于更难到达的区域。
3. 该 D4 结果应与 D1 灵敏度质量和 D3 分支可复现性一起解释，不能单独作为 IK 正确性结论。

## 六、文件索引

- `sample_plan.csv`：新增控制样本。
- `sample_results.csv`：每个样本的 FK 结果。
- `target_reachability_summary.csv`：每个目标的 D4 分类。
- `reachability_samples.npz`：新增样本点和目标点。
- `A1_sampled_reachability.png`、`A2_sampled_reachability.png`：三维采样可达性图。