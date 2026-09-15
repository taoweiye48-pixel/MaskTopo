# CrackForest MaskTopo 机制消融协议

冻结日期：2026-07-30  
状态：在查看本轮消融 test 结果前冻结

## 问题

MaskTopo 相对 grid 的约 18 pp 提升，究竟主要来自：

1. 稠密裂缝掩码监督提供的额外视觉信息；
2. 按预测连通分量进行的 token pooling；
3. 预测拓扑图上的消息传递；
4. 还是与图像对齐的真实拓扑组织？

## 固定设置

- 数据划分、裁剪、train/dev/test 样本和标签保持不变。
- 数据种子 `20260810`，优化种子 `20260810/11/12`。
- 每个种子使用对应的、已经训练完成的全局裂缝掩码预测器。
- 掩码阈值和闭运算次数读取既有 dev 选择结果，不重新查看 test 调参。
- 下游模型均为 16-token、`dim=64`、22 epochs。
- 除 `mask_feature_grid` 的第一层多一个输入通道外，其余方法使用相同参数量和训练过程。
- 本轮 test 已在此前方法开发中被查看，因此结果只作机制诊断，不作最终确认性证据。

## 方法

| 名称 | 图像输入 | token pooling | 图结构 |
|---|---|---|---|
| `grid_recheck` | 原图 + 两端点 | 固定 4×4 grid | 固定空间图 |
| `mask_feature_grid` | 原图 + 两端点 + 预测 mask 概率 | 固定 4×4 grid | 固定空间图 |
| `mask_assignment_identity` | 原图 + 两端点 | 预测分量保持 pooling | 单位图，无跨 token 消息 |
| `grid_mask_graph` | 原图 + 两端点 | 固定 4×4 grid | 由预测分量生成的图 |
| `shuffled_topology` | 原图 + 两端点 | 保持分量大小但随机打乱空间位置 | 打乱后的图 |
| `mask_topo_recheck` | 原图 + 两端点 | 预测分量保持 pooling | 预测分量图 |

`shuffled_topology` 对每个样本使用固定、与标签无关的随机排列；train/dev/test 不共享随机状态。

## 查看结果前固定的判断

拓扑特异性得到支持，需要同时满足：

1. `mask_topo_recheck - mask_feature_grid` 三种子平均至少 `+3 pp`，且三个种子方向均为正；
2. `mask_topo_recheck - shuffled_topology` 三种子平均至少 `+3 pp`，且三个种子方向均为正。

机制定位：

- `mask_topo_recheck - mask_assignment_identity ≥ 3 pp`：图消息传递有独立贡献；
- `mask_topo_recheck - grid_mask_graph ≥ 3 pp`：分量保持 pooling 有独立贡献；
- 若两者差值均小于 3 pp，只能说多种结构表示近似等效，不能把增益归因给某一个部件。

无论本轮结果如何，“通用连接器”仍需全新 untouched 数据集和更强 token-reducer 基线才能恢复。
