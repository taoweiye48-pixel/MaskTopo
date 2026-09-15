# MaskTopo DeepCrack 一次性外部确认协议

冻结日期：2026-07-30  
冻结时点：已完成数据格式与标签质量审计，尚未运行任何 DeepCrack test 性能实验

## 数据

- 官方仓库：`yhlleo/DeepCrack`
- 仓库提交：`8202a60701068645c883ed68b95fd3f30914d90c`
- 许可：仅限非商业科研与教学
- 官方 train：300 张；官方 test：237 张
- 官方 train 使用固定种子 `20260730` 拆为 240 train / 60 dev。
- 官方 237 张 test 不参与阈值、checkpoint、结构后处理或方法选择。

数据审计允许在冻结前检查文件数、尺寸、二值标签、前景比例和连通分量数量；不允许查看任何模型在官方 test 上的性能。

## 衍生任务

与 CrackForest 完全相同：

> 给定真实表面图像的 64×64 裁剪和两个可见端点标记，判断端点是否属于同一条人工标注裂缝连通分量。

- train/dev/test 样本数：2400 / 600 / 1200；
- 各 split 正负样本严格平衡；
- 原始图像按官方 split 和内部 dev split 隔离；
- 推理时不读取真实 mask；
- MaskTopo 与 mask-feature 控制均使用相同的预测 mask；
- 所有方法均压缩为 16 个 token。

## 方法

1. `grid_external`：普通固定网格；
2. `mask_feature_grid`：预测 mask 作为第四输入通道，仍为固定网格；
3. `mask_assignment_identity`：预测分量 pooling，无跨 token 图消息；
4. `grid_mask_graph`：固定网格 pooling，只加入预测图；
5. `shuffled_topology`：保持预测分量大小但打乱其空间位置；
6. `mask_topo_external`：预测分量 pooling + 对齐的预测图。

三个优化种子：`20260810/11/12`。每个种子独立训练全局 mask predictor 和下游模型；mask 阈值、形态学参数与 checkpoint 仅由 dev 选择。

## 冻结门槛

外部“拓扑特异性增益”确认成功，需要全部满足：

1. `mask_topo_external - grid_external`：平均至少 `+3 pp`，三个种子均为正；
2. `mask_topo_external - mask_feature_grid`：平均至少 `+3 pp`，三个种子均为正；
3. `mask_topo_external - shuffled_topology`：平均至少 `+3 pp`，三个种子均为正；
4. 三项按原始 test 图像聚类的 95% bootstrap CI 下界均大于 0。

“可靠恢复连通性”的更强命题保持原门槛：

- 每个种子的 test 直接结构 balanced accuracy 均至少 70%。

若前三项通过而结构门槛失败，只支持“拓扑监督的任务特定 token 路由”；不能恢复“通用、可靠的连通性保持 connector”表述。

## 数据审计修订记录

### v1：停止于数据门

首次候选集审计得到：

- endpoint-distance shortcut：54.00%，通过；
- mean-intensity shortcut：56.08%，超过 55%；
- 无模型在 DeepCrack 官方 test 上运行。

因此 v1 不进入训练，保留为失败的数据审计记录。

### v2：训练前冻结的匹配规则

为避免亮度成为标签捷径，v2 对每个 split：

1. 用固定种子生成目标样本数 3 倍的候选样本；
2. 使用候选样本的平均灰度和端点距离建立二维分位数箱；
3. 每个箱内等量抽取正负样本；
4. 从匹配对中固定种子抽取目标数量；
5. 不读取任何模型输出，不按 test 性能选择样本。

v2 必须重新通过原有四项数据门，才允许启动模型。此修订发生在任何 DeepCrack test 模型性能产生之前。
