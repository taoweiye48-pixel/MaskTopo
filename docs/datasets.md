# Dataset preparation

All four principal tasks classify endpoint connectivity from a 64×64 input with an image channel and two endpoint maps. Dense annotations define labels and supervise the image-only mask predictor. They are not supplied to the classifier at inference.

Download datasets from their original sources and follow their terms. Dataset image archives, full cached crops, and weights are not included here.

## FIVES

- Source: [Figshare dataset](https://doi.org/10.6084/m9.figshare.19688169.v1).
- Original release: 600 training and 200 test images.
- Manuscript source split: 480 training / 120 development; 198 test images contribute sampled pairs.
- Derived samples: 2,400 / 600 / 1,200.
- Preparation script: `experiments/prepare_fives_dataset.py`.

Extract the official archive so the preparation script can find `train/Original`, `train/Ground truth`, `test/Original`, and `test/Ground truth` (directory-name normalization handles spaces).

From `experiments/`:

```bash
python prepare_fives_dataset.py --source-root real_data/FIVES/dataset/extracted --output-root real_data/FIVES/dataset/preprocessed512_v2 --size 512
```

Use the ordinary green-channel preprocessing in `preprocessed512_v2`. The crop generator performs the single inversion. Masks use max pooling during downsampling to retain narrow vessels. Source split seed: 20260731; data seed: 20260810.

## DeepCrack

- Source: [official repository](https://github.com/yhlleo/DeepCrack).
- The upstream dataset restricts use to noncommercial research and educational purposes.
- Original split: 300 training / 237 test images.
- Manuscript source split: 240 training / 60 development; 222 test images contribute sampled pairs.
- Derived samples: 2,400 / 600 / 1,200.

Place `train_img`, `train_lab`, `test_img`, and `test_lab` below `experiments/real_data/DeepCrack/dataset/extracted/`, or pass the corresponding `--dataset-root`.

Source split seed: 20260730; data seed: 20260810. The historical external gate and the float16-aligned primary comparison are different protocols; see [reproduction.md](reproduction.md).

## Massachusetts Roads

- Source: [author's dataset page](https://www.cs.toronto.edu/~vmnih/data/).
- Source tiles: 1,108 / 14 / 49 training/development/test.
- Derived samples: 4,000 / 600 / 1,200.
- Budget: K=8; data seed: 20260802.

Expected layout under `experiments/real_data/MassachusettsRoads/` is `train/sat`, `train/map`, `valid/sat`, `valid/map`, and the corresponding test folders. The paired image and annotation stems must match. The training data gate uses 128×128 source crops resized to 64×64.

`prepare_massroads_dataset.py` provides source discovery and download options (`--help`). Run the data gate before training. Historical test evaluation is protected by a frozen manifest.

## RootNav2 Brassica

- Source: [author archive](https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip).
- Annotation format: RSML root polylines rendered by the preparation scripts.
- Source split after deduplication: 90 / 14 / 15.
- Derived samples: 2,400 / 600 / 1,200; K=16.
- Primary result uses the corrected matched-v3 endpoint-marker covariate protocol.

The sequence is `brassica_untouched_prepare.py` → `brassica_covariate_gate.py` → `brassica_train_dev.py` → `brassica_freeze_train_dev.py` → `brassica_test_once.py`. Original source-split and archive-index metadata are included under `experiments/实验记录/`. The archive index records file names, byte offsets and checksums, not image contents.

Some historical guards contain source-machine paths and frozen hashes. They intentionally reject mismatched artifacts. Generate matching local preparation outputs and inspect the corresponding protocol before training; do not disable guards to make an incompatible run appear to reproduce the original experiment. See the detailed limitations in [reproduction.md](reproduction.md).
