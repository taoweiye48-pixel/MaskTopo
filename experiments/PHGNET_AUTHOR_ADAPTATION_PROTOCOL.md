---
experiment_id: TB-B-260803-031
title: DeepCrack K16 PHG-Net author-code-based adaptation
status: frozen_before_test_run
dataset: DeepCrack
task: endpoint_connectivity
token_count: 16
optimization_seeds: [20260810, 20260811, 20260812]
split_seed: 20260730
primary_statistical_unit: source_image
confirmatory_status: retrospective_collision
---

# DeepCrack K=16 PHG-Net author-code-based adaptation protocol

## Question and claim boundary

This experiment tests whether MaskTopo's DeepCrack endpoint-connectivity result remains competitive against a stronger persistence-diagram encoder adapted from the PHG-Net authors' public implementation.

The method label is always **PHG-Net author-code-based adaptation**. It is not a reproduction of the original PHG-Net medical-classification protocol, and the original paper's accuracy values are not numerically compared with this endpoint task.

This experiment concerns endpoint connectivity only. It must not be mixed with the P1--P5 disease-classification result.

## Frozen author source

- Paper: Peng et al., *PHG-Net: Persistent Homology Guided Medical Image Classification*, WACV 2024.
- Repository: `https://github.com/yaoppeng/TopoClassification`.
- Frozen commit: `6daa5f7dba556e9882611eb4e2e1c89a67f0d2c5` (`master`, 2023-12-08).
- Reused module: `models/pointnet/pointnet_utils.py::PointNetEncoder`.
- Frozen module SHA256: `88535E17B99B6DE7BAD8CA4202698CBDBA1DD056CA1F0ADB67BD005729312B20`.
- The repository contains no explicit `LICENSE` file at the frozen commit. The clone is retained locally for research provenance; no license or redistribution permission is inferred.

## Frozen matched protocol

- Dataset/split: the same DeepCrack train/dev/test arrays used by TB-B-260802-022 and TB-B-260802-026: 2,400/600/1,200 crops, split seed 20260730.
- Statistical unit: source image; the 1,200 test crops map to 222 source images.
- Inputs: each optimization seed uses its already frozen saved mask-probability archive. The archive arrays must be stored as IEEE float16 before loading.
- Persistence data: reuse the frozen GUDHI cubical H0/H1 descriptors generated from `1 - predicted mask probability`, ranked by persistence. No ground-truth mask or topology is used at inference.
- Exact budget: the first 16 frozen persistence pairs are consumed and the adapter must emit exactly 16 token vectors for every image.
- Training: 22 epochs, batch 32, AdamW, learning rate `1e-3`, weight decay `1e-4`, mixed precision, the same augmentation, optimization seeds 20260810/11/12, and development-only checkpoint selection.
- Head: reuse `CommonTokenClassifier` and its `CoarseGraphHead` unchanged.
- Test access: the DeepCrack test has already been observed. Each seed is evaluated once only after its development-selected checkpoint is restored. The experiment is therefore a retrospective collision, not a new untouched confirmation.

## Author-code adapter

1. Each selected PD point is represented as `(birth, death, H0 one-hot, H1 one-hot)`, matching the four-channel PD representation described in the paper. DeepCrack K=16 has zero padding in all three frozen seeds.
2. Apply the author's point-cloud centering and maximum-radius normalization per image.
3. Call the unmodified author `PointNetEncoder(global_feat=False, feature_transform=False, channel=4)`. Its output concatenates the repeated 2,048-dimensional global PD feature with the 64-dimensional point feature, retaining one output per input point.
4. Project the resulting 2,112-dimensional features to the common token dimension. Assert output shape `[batch, 16, dim]`.
5. Apply the author repository's residual channel-gating pattern to the common `PatchStem` visual context: `context + context * sigmoid(W(global_PD))`.
6. Use the same two visual-interaction blocks as the existing PH-guided baseline to isolate the changed PD encoder within the common fixed-K interface.
7. Send the 16 final tokens, persistence mass, and attention-derived centroids to the unchanged common classification head.

These steps are necessary task/interface adaptations. They are reported in full and must not be described as the original PHG-Net architecture or protocol.

## Comparators and frozen analysis

- Float16-aligned MaskTopo predictions from TB-B-260802-026.
- Existing PH-only and PH-guided predictions from TB-B-260802-022.
- PHG-Net author-code-based adaptation from this experiment.

All four methods are reported separately for each seed and as mean plus sample standard deviation. No best-method selection is used to hide an unfavorable result.

## Statistical analysis

- Primary contrast: float16-aligned MaskTopo minus PHG-Net author-code-based adaptation.
- Secondary contrasts: adaptation minus PH-only and adaptation minus PH-guided.
- Resampling: paired source-image cluster bootstrap, 20,000 repetitions, statistics seed 20260803.
- Report point estimates and two-sided percentile 95% confidence intervals.
- A positive primary interval supports only the narrow claim that MaskTopo outperforms this author-code-based adaptation on the already observed DeepCrack endpoint task.
- If the primary interval crosses zero, the comparison is inconclusive.
- If the adaptation has a positive interval over MaskTopo, any DeepCrack claim of superiority to stronger PH encoders is withdrawn; the result remains informative regardless of direction.

## Execution gates

1. Verify author commit, module hash, clean clone, CUDA availability, float16 archives, PH-cache hashes, exact K, and zero test access during self-test.
2. Run architecture/gradient self-test without DeepCrack test evaluation.
3. Run seed 20260810. A crash is reported and not automatically retried.
4. If the run completes normally, run seeds 20260811 and 20260812 sequentially.
5. Validate stored predictions against recorded metrics and then run the frozen cluster-bootstrap analysis.

## Required records

- Runtime stdout/stderr logs under `E:\TopoBridge_MVP\实验记录`.
- Per-seed checkpoints, predictions, histories, and `result.json` under the matching `results_phgnet_author_deepcrack_k16_seed*` directory.
- Final experiment report and artifact hashes under `E:\TopoBridge_MVP\实验记录`.

