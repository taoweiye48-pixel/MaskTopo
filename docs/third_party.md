# External sources and attribution

The source release does not vendor the upstream repositories below. In-framework baseline implementations and adaptations are identified in the manuscript and original protocol files; they should not be presented as exact executions of an upstream training pipeline.

## Baseline author repositories

- PHG-Net adaptation: [yaoppeng/TopoClassification](https://github.com/yaoppeng/TopoClassification), inspected checkout `6daa5f7dba556e9882611eb4e2e1c89a67f0d2c5`. `phgnet_author_adaptation.py` loads `models/pointnet/pointnet_utils.py` at runtime. Obtain the upstream source separately and pass `--author-repo`, or place it at `experiments/third_party/TopoClassification`. Consult the upstream terms before reuse.
- G2TM reference: [vbercy/g2tm-segmenter](https://github.com/vbercy/g2tm-segmenter), inspected checkout `f17d1c8374a6f09365d856ff40d5aaf6d0bcf5d4` on `torch2`. The manuscript uses an in-framework matched adaptation for endpoint classification; this is not a claim to reproduce the upstream semantic-segmentation benchmark.

## Datasets and figures

- FIVES: [Jin et al. dataset](https://doi.org/10.6084/m9.figshare.19688169.v1), CC BY 4.0 as recorded with the data release. The overview's retinal input and component illustrations derive from development example #232, source `train:74_A`. Colors, partitions, feature visualization and graph diagrams were created for this manuscript. Raw source images are not otherwise included.
- DeepCrack: [author repository](https://github.com/yhlleo/DeepCrack); its dataset is restricted by its owners to noncommercial research and educational purposes.
- Massachusetts Roads: [author dataset page](https://www.cs.toronto.edu/~vmnih/data/).
- RootNav2: [Brassica source archive](https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip). Consult the original dataset publication and distribution terms.

The original experiment files cite additional methods and libraries. Their inclusion in this repository does not change external dataset or software licenses.
