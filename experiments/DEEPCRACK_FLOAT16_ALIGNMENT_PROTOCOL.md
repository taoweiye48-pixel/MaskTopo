---
date: 2026-08-02
experiment_id: TB-B-260802-026
title: DeepCrack K16 float16 probability-path alignment counterfactual
status: frozen_before_implementation_and_execution
route: route_2_topology_preservation_efficiency
---

# DeepCrack K=16 float16 路径对齐反事实协议

## 冻结问题

历史 DeepCrack MaskTopo 使用 mask predictor 的 in-memory float32 概率构图，而 PH/reducer 使用保存后再读取的 float16 概率缓存。该实验只回答：当 MaskTopo 改为消费与 PH 完全相同的 float16 缓存值时，MaskTopo 相对 PH 的端点连通优势是否仍存在。

## 冻结范围

- 数据：DeepCrack 已观察的冻结 test，1200 crops、222 个 source images。
- K：16。
- 优化种子：20260810、20260811、20260812。
- MaskTopo checkpoint、PH-only/PH-guided predictions、分类阈值和所有模型参数保持冻结。
- 不训练、不选择 checkpoint、不调整 mask threshold、不调 closing。
- Massachusetts Roads official test 不读取、不推断、不重算。
- P1-5 疾病分类结果不使用。

## 唯一干预

1. 从每个冻结 DeepCrack run 的 `mask_probabilities.npz` 读取 `test`。文件中的原始 dtype 必须为 float16。
2. 允许为 NumPy/PyTorch 运算把缓存值转换为 float32，但不得恢复量化前信息；记录原始 float16 字节哈希与量化值哈希。
3. 使用原冻结参数 `threshold=0.90`、`closing=0`、`K=16`、`mode=mask_topo_recheck`、artifact seed `20260731` 重新构造 MaskTopo assignment、adjacency 和 reachability。
4. 使用原冻结 MaskTopo checkpoint 重新前向；分类阈值固定为 0.5。
5. PH-only 与 PH-guided 使用既有保存 predictions，禁止重新训练或按本轮结果选择模型。

## 强制复现与一致性门

- truth、source_id 在 MaskTopo 原结果与 PH 结果之间逐元素一致。
- 原生 float32 MaskTopo prediction 必须逐元素复现其已报告 balanced accuracy。
- PH-only/PH-guided predictions 必须逐元素复现 TB-B-260802-022 已报告 balanced accuracy。
- 三个 MaskTopo checkpoint 哈希必须记录。
- 对齐前后记录 assignment、adjacency、reachability 的样本级变化数和二分类 prediction flip 数。

## 冻结统计

- 每种子报告 native-float32 与 aligned-float16 MaskTopo balanced accuracy。
- 报告 aligned MaskTopo − PH-guided、aligned MaskTopo − PH-only，以及 aligned MaskTopo − 冻结 best-PH（DeepCrack 已冻结为 PH-only）。
- 以 source image 为 cluster，跨三个优化种子平均差，20,000 次配对 bootstrap；统计种子 20260826。
- 同时对 `aligned − native` 做同一来源级 bootstrap，量化精度路径本身造成的变化。

## 冻结裁决规则

- 若 aligned MaskTopo − PH-only 的 95% source-cluster bootstrap CI 下界仍大于 0，则精度差异没有解释掉 DeepCrack 的 PH 优势结论；以 aligned 数值替换该对照的公平性敏感性报告，但不覆盖历史原生路径结果。
- 若 CI 跨零，则 DeepCrack 相对 PH 的证据降级为不确定。
- 若点估计小于等于 0，则撤回 DeepCrack 上优于 PH 的主张。
- 无论结果如何，历史 float32/float16 不一致仍作为 provenance limitation 保留；本实验只消除其对结论的混杂，不改写历史执行事实。

## 产物

- `results_deepcrack_float16_alignment/PROTOCOL.md`
- `results_deepcrack_float16_alignment/result.json`
- `results_deepcrack_float16_alignment/REPORT.md`
- `results_deepcrack_float16_alignment/source_data.csv`
- `results_deepcrack_float16_alignment/predictions_seed*.npz`
- `实验记录/TB-B-260802-026_DeepCrack_float16路径对齐.md`
- 正式 stdout/stderr 日志及代码/协议/checkpoint/cache SHA-256。

