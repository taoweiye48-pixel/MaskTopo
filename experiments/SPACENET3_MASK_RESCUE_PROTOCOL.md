---
date: 2026-08-03
experiment_id: TB-B-260803-029
title: SpaceNet 3 Paris sparse-road mask predictor rescue gate
status: frozen_before_implementation_and_execution
parents: [TB-B-260802-027, TB-B-260803-028]
---

# SpaceNet 3 稀疏道路 mask predictor 修复门协议

## 问题

TB027 的固定 0.5 mask 前景率为 35.74%，而 dev GT 道路像素率为 1.29%；TB028 表明阈值校准不能同时恢复 road F1 与路径 proxy。本实验仅在 Paris train/dev 上比较三种预列稀疏分割损失，判断是否值得继续 K=16 自然图 decoder 开发。

## 固定数据与模型

- 复用 TB027 冻结的 2400 train / 800 dev crops 和 source_id；不访问 Khartoum。
- 所有候选使用同一 RoadUNet、同一 train augmentation、batch=64、AdamW lr=1e-3、weight decay=1e-4、20 epochs、seed=20260840。
- checkpoint 只按各自 dev training loss 最低选择。
- 候选损失：
  1. `pos8_dice`：标准逐像素 BCEWithLogits，positive weight=8，加 soft Dice；
  2. `focal_dice`：alpha=0.75、gamma=2 的 binary focal loss，加 soft Dice；
  3. `pos4_dice_cldice`：positive weight=4 BCE，加 soft Dice，加 0.5× differentiable soft-clDice loss。
- 保存 dev 概率为 float16 后重读，再进行所有阈值和指标计算。

## 冻结选择与指标

- 阈值候选：`0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.925,0.95`。
- 每个候选按 source-tile mean road F1 选阈值；并列取更高阈值。选阈值时不查看 raster-APLS。
- 三个 loss 候选同样按 selected source-mean road F1 选择 best；并列依次取更低前景率、候选列表靠前者。
- best 冻结后报告 clDice、raster-APLS proxy、path recall、前景率和有效 crop 数；三个候选全部报告。

## 继续门

best candidate 必须同时满足：

- source-mean road F1 `>=0.15`；
- source-mean raster-APLS proxy `>=0.20`；
- 预测前景率 `<=0.05`。

通过只授权新的 Paris train/dev decoder 开发，不直接授权 held-out。失败则当前低分辨率自然道路扩展暂停，需更强/更高分辨率分割器后再立新协议。

## 主张边界

本实验是开发门，不是 test，不改变 TB027 FAIL，不影响已有端点连通证据，不使用 P1-5 疾病分类结果。

