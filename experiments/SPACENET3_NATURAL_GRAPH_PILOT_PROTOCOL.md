---
date: 2026-08-02
experiment_id: TB-B-260802-027
title: SpaceNet 3 city-held-out natural road-graph utility pilot
status: frozen_before_manifest_and_heldout_content_access
route: route_2_topology_preservation_efficiency
---

# SpaceNet 3 城市外推自然道路图效用 pilot 协议

## 冻结问题与主张边界

本实验只回答：在同为 `K=16` 的压缩预算和同构空间解码器下，由预测道路掩膜构造的 MaskTopo token，是否比通用 token reducer、mask 信息对照和 persistent-homology token 更好地保留自然道路图的路径结构。

它不是疾病分类实验，不使用、合并或重新解释 P1-5 的 `+1.00 pp`；也不是端点二分类实验，不能把此前端点连通任务的大幅 balanced-accuracy 增益直接写成这里的效应。本实验不读取或重算 Massachusetts Roads official test。

这是选定子集、低分辨率 raster-to-graph 的城市外推 pilot；除非以后在完整 SpaceNet 官方 APLS 管线复现，否则不得称为完整 SpaceNet leaderboard 结果。

## 数据、许可与 sealed split

- 数据：SpaceNet 3 Roads，官方公开 S3 的 `PS-RGB` 影像和 `geojson_roads` 中心线。
- 许可：按 SpaceNet 官方数据页披露的 CC BY-SA 4.0 使用；下载清单保留对象 key、大小、ETag 和哈希。
- train：Paris，96 个成对源瓦片。
- dev：Paris，与 train 不重叠的 32 个成对源瓦片。
- held-out test：Khartoum，64 个成对源瓦片；在本协议和对象清单冻结前禁止读取影像或标签内容。
- 选样不看内容：对成对对象 ID 按 `SHA256("TB-B-260802-027|<city>|img<ID>")` 升序。Paris 前 96 个为 train、随后 32 个为 dev；Khartoum 前 64 个为 held-out test。
- 清单一经生成不得覆盖；对象元数据、协议和清单 SHA-256 均归档。

## 固定样本生成

- 每个 1300×1300 源瓦片固定取 25 个不重叠 256×256 窗口；行列起点均为 `[0,256,512,768,1024]`，边缘剩余 20 像素不使用。
- PS-RGB 固定以 `clip(value / 2047, 0, 1)` 归一化；不使用 test 统计量。
- GeoJSON 中心线以影像 CRS 栅格化，`all_touched=true`，原生 256×256 二值图经 4×4 max-pooling 得到 64×64 标签。
- 不得根据空道路、连通分量、困难程度或任何标签统计过滤窗口；必须报告空道路比例。
- 增强仅限 train 的水平翻转、垂直翻转、90° 旋转和固定范围亮度/对比度；dev/test 不增强。

## 固定模型和训练机会

- token budget：`K=16`；特征维度 64；64×64 输入经同构 PatchStem 得到 16×16 细粒度特征。
- 所有 token 方法使用同构空间解码器：坐标查询与 K tokens 做两层 cross-attention，再用两级上采样输出 64×64 道路 logit。各方法单独训练权重，但架构和训练预算相同。
- 方法全集：`grid`、`tokenlearner`、`perceiver_resampler`、`tome_style`、`mask_guided_queries`、`ph_only`、`ph_guided`、`mask_assignment_identity`、`shuffled_topology`、`mask_topo`。
- PH 描述子：复用冻结的 GUDHI cubical-complex 实现；对 `1 - predicted road probability` 计算 H0/H1，按 persistence 排序取 16 个，保持 PH-only 与 PH-guided 的语义区分。
- MaskTopo、assignment、shuffled 和 mask-guided 共用同一优化种子的冻结 mask predictor 概率；mask threshold 固定 0.5、closing 固定 0，不做 test 调参。
- `mask_only` 直接预测结果作为非 token 压缩参考上界/诊断，不参加“同为 K=16”的主排名。
- 优化种子：`20260830, 20260831, 20260832`；统计 bootstrap 种子 `20260833`。
- mask predictor：12 epochs；token 模型：12 epochs；AdamW，lr=1e-3，weight decay=1e-4，batch=64；损失为 balanced BCE + soft Dice。
- checkpoint 只按 dev loss 最低选择；二值阈值固定 0.5，不按 APLS 或 test 选择模型。

## 分阶段门控与一次性测试

1. 数据门：对象完整、图像可读、标签可投影、固定窗口数量正确；只在 Paris 内容上完成实现/自检。
2. 单种子 dev 机制门：种子 20260830。若 `mask_topo` 的 symmetric raster-APLS proxy 不同时满足：
   - 比 strongest generic reducer 高至少 0.03；
   - 高于 `mask_assignment_identity`；
   - 高于 `shuffled_topology`；
   则停止，不读取 Khartoum 内容，并如实报告不可行。
3. 若通过，补齐三种子 train/dev，冻结所有 checkpoint、预测阈值、代码哈希和 dev 选出的 strongest generic reducer。
4. test 只运行一次；任何失败都另建 retry 产物，禁止覆盖或按 test 结果调参。

## 冻结指标

- 主指标：symmetric raster-APLS proxy。它遵循 APLS 的归一化最短路径长度误差和缺路记零思想，但由 64×64 skeleton graph 与固定 3 像素 snap tolerance 构造；必须明确标注为 proxy，不能冒充官方 SpaceNet APLS。
- 同时报告：road pixel F1、soft Dice、clDice、路径召回率、断连率、空 GT 窗口比例。
- 对每个方法报告参数量、部分 FLOPs、batch=1/8/32 latency；mask/PH CPU 构图成本单独列出和纳入端到端说明。
- 主比较：MaskTopo 减去三种子 dev 预选出的 strongest generic reducer。
- 预列次比较：MaskTopo 减去 PH-only、PH-guided、assignment、shuffled；全部报告，不以显著性筛选。
- 推断单位为 source tile；跨三个优化种子先取同一源瓦片均值，再做 20,000 次 paired source-cluster bootstrap，报告 95% CI。

## 主张规则

- 只有 held-out 主比较点估计为正且 95% CI 下界大于 0，才能写“在该冻结 SpaceNet 3 城市外推 pilot 上保持自然道路路径优于 strongest generic reducer”。
- PH 或机制对比未通过时必须相应降级“拓扑机制优于 PH/assignment/shuffled”的表述。
- 无论结果如何，都只属于自然道路图重建，不与疾病分类 `+1 pp` 或端点连通 balanced accuracy 合并。

