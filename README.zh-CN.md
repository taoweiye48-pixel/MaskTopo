# MaskTopo

[English](README.md) | **简体中文**

**面向细长结构连通性的连通分量对齐 Token 粗化**

本仓库提供论文 **MaskTopo: Component-Aligned Token Coarsening for Thin-Structure Connectivity** 的配套代码。

MaskTopo 根据预测的连通分量，将密集视觉特征组织为**恰好 K 个非空 token**，再通过分别设置门控的**邻接**与**可达性**分支传递信息。

![MaskTopo 整体流程](assets/overview.png)

## 方法

- **CAC — Component-Aligned Coarsening（连通分量对齐粗化）：**在保留的连通分量之间分配 token 预算，对每个分组进行划分，并聚合外观特征以及质心、支持区域大小等元信息。
- **GTM — Gated Topological Messaging（门控拓扑消息传递）：**通过独立门控和残差连接，组合基于行归一化邻接矩阵与可达矩阵的消息。
- **连通性分类头：**使用两层、四头 Transformer，预测两个标记端点是否连通。

分类器输入形状为 `[B, 3, 64, 64]`，包含一个图像通道和两张端点标记图。掩码预测器仅接收图像通道，输出结构概率。**真实标注掩码用于训练监督和定义连通性标签，推理时不作为模型输入。**

<details>
<summary>模块示意图</summary>

![连通分量对齐粗化 CAC](assets/cac.png)

![门控拓扑消息传递 GTM](assets/gtm.png)

</details>

## 安装

```bash
git clone https://github.com/taoweiye48-pixel/MaskTopo.git
cd MaskTopo
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

运行原实验脚本时，安装实验依赖：

```bash
python -m pip install -e ".[experiments]"
```

发布验证使用 Python **3.12.10**、PyTorch **2.11.0+cu128**、NumPy **2.3.5** 和 SciPy **1.16.1**。核心测试在 CPU 上运行。请根据本机环境选择合适的 PyTorch 版本；完整版本记录见[环境文件](requirements-tested.txt)。

## 快速开始

```bash
python examples/smoke_demo.py
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
```

示例使用合成输入和随机权重检查接口，不用于复现训练后的准确率。测试覆盖精确 token 数量、连通分量边界、图可达性、零初始化门控、权重文件兼容性，以及与原实现的数值一致性。

### 使用预测掩码

```python
import torch
from masktopo import MaskTopo, MaskPredictor, build_topology

model = MaskTopo(token_count=8, dim=64).eval()
predictor = MaskPredictor().eval()
# 正式评估前，分别为分类器和掩码预测器加载训练好的 state_dict。

images = torch.rand(2, 3, 64, 64)  # 替换为图像通道和两张端点标记图。
with torch.no_grad():
    probabilities = predictor(images[:, :1]).sigmoid().squeeze(1)
    topology = build_topology(
        probabilities,
        token_count=8,
        threshold=0.5,       # 此处仅为示例；正式取值应在 development 数据上选择。
        closing_iterations=0,
    )
    logits = model(images, *topology.tensors(device=images.device))
    connected_probability = logits.sigmoid()
```

当 `dim=64` 时，K=8 和 K=16 的 `MaskTopo` 分类器均有 **149,123 个参数**，不包括独立的掩码预测器。公开模型保留了原分类器的 `state_dict` 键名。`build_topology` 接受 NumPy 数组或已分离梯度的张量，并在 CPU 上执行离散的连通分量分配。

## 端点连通性实验结果

**所有报告结果均以当前论文为准。** 下表对应论文的主要实验协议，MaskTopo 列为三个随机种子的平衡准确率（BA，%）均值 ± 标准差：

| 数据集 | K | MaskTopo BA | 主要比较方法 | 比较方法 BA | 提升（pp） |
|---|---:|---:|---|---:|---:|
| FIVES | 8 | 89.36 ± 2.79 | TokenLearner | 54.61 | +34.75 |
| DeepCrack | 16 | 74.97 ± 2.60 | Mask-Perceiver | 66.78 | +8.19 |
| Massachusetts Roads | 8 | 76.94 ± 0.76 | Matched G2TM | 67.14 | +9.80 |
| RootNav2 Brassica | 16 | 89.14 ± 0.81 | Mask-Slot | 69.53 | +19.61 |

pp 表示百分点。以上均为**端点连通性分类**结果，不同结构领域的比较方法和评估协议有所区别。DeepCrack 的主要结果使用各方法共享的、以 float16 存储的预测掩码概率；历史 float32 消融属于另一组实验。Massachusetts 和 Brassica 的对应关系消融使用事后开展的 development 数据分析。详见[实验协议说明](docs/reproduction.md)和[论文表 1–4 的 CSV 文件](reported_results/README.md)（英文）。

## 实验复现

先完成[数据准备](docs/datasets.md)，再按照[实验指南](docs/reproduction.md)运行（上述详细文档为英文）。

仓库包含原训练、评估和数据处理脚本，以及实验协议文档。数据集、完整样本缓存和训练权重需要另行获取或生成。本次代码整理与发布没有重新执行完整数据集训练。

| 路径 | 内容 |
|---|---|
| `masktopo/` | 可安装的 CAC、GTM、分类器和掩码预测器接口 |
| `experiments/` | 原训练、评估、消融、数据准备和结果汇总脚本 |
| `examples/` | 可直接运行的小型接口示例 |
| `tests/` | 结构约束检查及与原实现的数值一致性测试 |
| `reported_results/` | 按论文表 1–4 导出的汇总结果 |
| `docs/` | 数据集链接、实验协议、来源记录和外部依赖说明 |
| `assets/` | 整体流程图与模块图 |

`topocoarsen_oracle.py` 等早期文件名保留了项目开发过程中的命名。在 MaskTopo 流程中，该文件的分配函数接收的是**预测连通分量标签**，文件名不表示推理时使用真实标注。

## 代码来源与外部组件

[original_source_sha256.json](docs/original_source_sha256.json) 记录原始代码文件的哈希值；[source_changes.md](docs/source_changes.md) 说明发布版本对结果说明文字的修改；[release_source_sha256.json](docs/release_source_sha256.json) 记录发布文件的哈希值。公开接口中的神经网络模块提取自这些实验脚本。一致性测试在共享权重下比较分配结果、图矩阵、分类输出和输入梯度，并覆盖非零图门控的情况。

示例图使用 FIVES development 样本 #232（`train:74_A`）。原始图像来自 Jin 等人的 [FIVES 数据集](https://doi.org/10.6084/m9.figshare.19688169.v1)，按 CC BY 4.0 许可发布。图中的特征、分区和图结构是由该样本生成的说明性可视化，不构成额外的性能实验。详见[外部来源说明](docs/third_party.md)（英文）。

## 引用

在作者信息和公开论文标识补全之前，可按论文标题引用代码仓库，并注明所使用的具体提交版本：

```bibtex
@misc{masktopo_code,
  title = {MaskTopo: Component-Aligned Token Coarsening for Thin-Structure Connectivity},
  year = {2026},
  howpublished = {\url{https://github.com/taoweiye48-pixel/MaskTopo}},
  note = {Companion source code}
}
```

## 许可证

本项目尚未指定统一的软件许可证。数据集和上游代码各自适用原作者的许可条款，详见[外部来源说明](docs/third_party.md)（英文）。
