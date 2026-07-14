# A1/A2 有监督逆运动学成功台账重分类记录

## 一、目的

本轮审计用于把历史正式实验和诊断实验中的成功结果重新分账，区分冷启动、失败后冷重启、邻点延拓、分支状态重放、零迭代记录验证和阶段辅助结果，避免把路径依赖结果误写成独立冷启动算法成功。

## 二、输入范围

- 扫描 summary.csv 数量：1
- 结果行数量：40

## 三、总览结果

- 原始 5 mm 成功行数：11
- 严格冷启动成功行数：0
- 失败后冷重启成功行数：2
- 邻点或旧成功点延拓成功行数：8
- 分支状态重放成功行数：0
- 后验分支回环停滞行数：0

分类计数：

- `cold_restart_after_failure`：5
- `cold_start`：2
- `manual_initial_state`：1
- `neighbor_continuation`：10
- `stale_success_continuation`：22

## 四、解释原则

1. 只有未加载外部分支状态、初值来源为标准冷启动，并且真实末端误差进入 5 mm 的结果，才计入严格冷启动成功。
2. `last_success_*` 和 `last_success_before_failed_point` 只证明邻点延拓或路径依赖有效，不证明独立点冷启动可达。
3. 加载历史分支状态得到的结果只说明该平衡分支存在，不能计入冷启动成功率。
4. 缺少关键字段的旧格式结果被标为证据不足或旧格式推断，不作为严格成功证据。

## 五、输出文件

- `success_ledger.csv`：逐点重分类台账。
- `success_ledger_summary.csv`：按实验和导管类型聚合的摘要。
- `success_ledger_summary.json`：机器可读总览。
- `success_ledger_overview.png/svg`：成功证据类别概览图。
