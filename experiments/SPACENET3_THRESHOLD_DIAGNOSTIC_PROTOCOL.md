---
date: 2026-08-03
experiment_id: TB-B-260803-028
title: SpaceNet 3 Paris dev threshold recoverability diagnostic
status: frozen_before_execution
parent: TB-B-260802-027
---

# SpaceNet 3 稀疏道路阈值可恢复性诊断协议

## 目的

只用 TB-B-260802-027 已保存的 Paris dev float16 概率，判断固定阈值 0.5 是否是 mask-only 与 K=16 decoder 严重前景过预测的主要原因。该诊断不能改写 TB027 的失败门，也不能授权访问 Khartoum held-out。

## 冻结输入与方法

- 输入：TB027 seed 20260830 的 11 份 dev probability、冻结 dev cache、source_id/crop 坐标。
- 方法：mask-only、Grid、TokenLearner、Perceiver、ToMe、mask-guided、PH-only、PH-guided、assignment identity、shuffled、MaskTopo；全部报告。
- 候选阈值：`0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99`。
- 每个方法按 32 个 source tile 的 road F1 均值选择阈值；并列时取更高阈值以减少稀疏道路任务的假阳性。
- 选阈值时不计算或查看 raster-APLS；阈值冻结后才计算一次 clDice、raster-APLS proxy、path recall 和 disconnection rate。
- 双空 crop 的 F1/APLS 记 NaN，不作为奖励；所有 crop 保留，报告有效 crop 数和预测前景率。

## 诊断门

若 mask-only 同时满足：

1. selected road F1 比阈值 0.5 提高至少 `0.05`；
2. selected raster-APLS proxy 比阈值 0.5 提高至少 `0.03`；
3. selected 前景率不高于 `0.10`；

则判为 `CALIBRATION_MATERIAL`，下一轮可在 Paris train/dev 上预注册校准后的 artifact 与 decoder 开发。否则判为 `RETRAIN_MASK_REQUIRED`。无论判定为何，Khartoum 都保持 sealed；新的 held-out 访问必须由新的三种子 dev gate 单独授权。

## 主张边界

这是失败诊断，不是 confirmatory test，不产生新的论文性能主张，不与端点连通或 P1-5 疾病分类结果合并。

