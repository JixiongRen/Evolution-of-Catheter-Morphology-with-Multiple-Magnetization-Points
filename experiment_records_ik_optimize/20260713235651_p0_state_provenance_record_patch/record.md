# P0 状态来源与门控口径记录修正

## 一、目的

本轮针对 A1/A2 有监督逆运动学诊断审计中的 P0 问题进行修正。核心目标是使后续灵敏度审计和外层优化器回归能够明确区分状态来源、阶段门控和控制通道门禁，避免把不同平衡状态或不同实验口径下的结论混用。

## 二、修改内容

修改文件：

`DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`

主要变化：

- 在点级摘要中增加状态来源记录，包括最终状态、最后接受状态和初始状态的状态哈希、热启动哈希、中心线哈希和状态文件路径。
- 在点级摘要中增加分支状态元数据，包括代码提交、正运动学配置哈希、几何哈希、材料哈希、磁体哈希、求解器哈希和灵敏度配置哈希。
- 在点级摘要中增加运行设备字段，用于确认实验使用 GPU。
- 在 `summary.csv` 中增加上述关键字段，便于后续自动筛选和审计。
- 将中文报告中的“门控”拆分为“阶段门控”和“控制通道门禁”两个口径。
- 修正强制子集实验报告中容易误读的“未使用门控阻断”表述，明确其含义是“不使用代表点结果作为阶段阻断”，而不是“未启用控制通道门禁”。
- 在批次元数据中增加实际推进搜索边界、静态物理长度上限、阶段门控标志和控制通道门禁说明。

## 三、验证

已完成以下静态验证：

- 使用项目 conda 环境执行脚本编译检查。
- 导入脚本并验证批次控制通道门禁中文标签函数。
- 从既有 A1-pt01 实验目录中读取 `run_config.json`，确认能提取最终状态来源、状态哈希、中心线哈希、代码提交和 GPU 信息。

验证命令：

```bash
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py
```

```bash
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location('formal', Path('DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py'))
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
point = Path('DELIVER_TO_EXP/formal_experiments/20260713232320_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_01')
run_config = mod._read_json(point / 'run_config.json', {})
prov = mod._state_provenance_summary(point, run_config)
print(prov['selected_state_source'], prov['selected_state_source_zh'])
print(prov['selected_state']['z_star_hash'][:12])
print(prov['selected_state']['centerline_hash'][:12])
print(prov['branch_metadata']['code_commit'][:12])
print(prov['runtime_environment']['devices'][0]['platform'])
PY
```

关键输出：

```text
final 最终状态
9adfdeecb9ef
54c713226eeb
676af5b276e6
gpu
```

## 四、结论

本轮只完成记录语义和证据链修正，不改变正运动学物理模型、空间离散、容差、真实末端定义或控制边界。后续 P1 灵敏度审计可以直接使用新增的状态来源字段，避免把失败态、最后接受态、最终态和新运行态混为同一类证据。

## 五、下一步

下一步应启动 A1 三点门控与步长解耦实验，优先验证：

- 强门控失败主要来自小步长，还是来自控制子空间过窄。
- A1-pt01 在分层策略下是否保持 5 mm 内成功。
- A1-pt13 是否能通过全局入盆加局部精修进入 5 mm。

