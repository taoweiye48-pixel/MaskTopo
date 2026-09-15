---
date: 2026-08-02
experiment_id: TB-B-260802-024
title: Frozen K16 threshold, morphology, and topology-stress audit
status: frozen_before_implementation_and_execution
route: route_2_topology_preservation_efficiency
---

# Frozen K=16 robustness protocol

## Scope

- Datasets: CrackForest and DeepCrack.
- Optimization seeds/checkpoints: 20260810, 20260811, 20260812.
- Token budget: K=16, the primary matched direct-comparison setting.
- Models: MaskTopo, TokenLearner, Perceiver Resampler, ToMe-style, mask-guided queries, PH-only, and PH-guided.
- No training, checkpoint selection, threshold selection, or classifier recalibration is permitted.
- Classification threshold remains 0.5.
- Tests were previously observed; the entire audit is retrospective robustness evidence.
- Massachusetts Roads remains sealed.

## MaskTopo threshold and morphology sensitivity

Use clean frozen predicted-mask probabilities and the same frozen MaskTopo checkpoint. The one-factor conditions are fixed before execution:

- threshold sweep at closing=0: 0.80, 0.85, 0.90, 0.95;
- morphology sweep at threshold=0.90: closing iterations 0, 1, 2.

The development-selected reference is threshold=0.90, closing=0. No test result may select a preferred alternative setting.

Precision audit: the original MaskTopo mechanism runs constructed topology from the mask predictor's in-memory float32 output and then saved probabilities as float16, whereas later reducer/PH runs consumed the saved float16 cache. Robustness must preserve each clean frozen native path and report the float32–float16 discrepancy. It may not silently substitute one path and call mismatched predictions a checkpoint failure.

## Stress conditions

All perturbations are deterministic, fixed before execution, and use no ground-truth mask, connectedness label, or endpoint location.

1. `clean`: unchanged image and predicted-mask probability.
2. `mask_break_erosion1`: threshold the predicted mask at 0.90, erode once with a 3×3 structure, and set only removed foreground probabilities to zero. The image is unchanged.
3. `mask_false_link_closing2`: threshold at 0.90, apply two 3×3 binary closings, and set only newly added pixels to probability one. The image is unchanged.
4. `low_contrast_0.35`: replace gray channel x by `0.5 + 0.35*(x-0.5)`, clip to [0,1], preserve endpoint-marker channels, and rerun the seed-specific frozen mask predictor.
5. `cross_domain_noise`: replace gray channel x by `0.85*x + 0.075 + Normal(0,0.08)`, clip to [0,1], preserve endpoint-marker channels, and rerun the seed-specific frozen mask predictor. The noise field uses input seed 20268024 plus a fixed dataset offset and is identical across optimization seeds.

Unmasked reducers reuse clean predictions for the two mask-only corruptions but are re-evaluated for both image perturbations. Mask-guided, MaskTopo, PH-only, and PH-guided are re-evaluated whenever mask probability changes.

## Measurements

- endpoint-connectivity balanced accuracy by dataset, seed, model, and condition;
- drop from each model's clean result;
- MaskTopo minus frozen clean-strongest general reducer;
- MaskTopo minus PH-guided (primary PH comparator) and minus the descriptive best-PH envelope;
- source-image cluster bootstrap 95% confidence intervals, 20,000 repetitions;
- predicted-mask pixel F1, foreground fraction, and changed-pixel fraction;
- MaskTopo direct structural balanced accuracy and graph-density diagnostics;
- H0/H1 composition of PH tokens;
- descriptive repeated-measures-centered association between mask F1 and MaskTopo downstream gain.

The fixed clean-strongest general reducers are TokenLearner for DeepCrack and Perceiver Resampler for CrackForest, as established before this robustness execution by the verified K=16 Pareto summary.

## Reporting boundaries

- Report every condition and failure; do not choose a favorable perturbation after test observation.
- PH-only and PH-guided remain separate. Any best-PH envelope is explicitly descriptive.
- Mask F1/downstream associations are exploratory repeated-measures diagnostics, not independent-sample causal evidence.
- Grid, Assignment, Shuffled, G2TM, TopoCL, and WACV 2026 scores are not fabricated into this unmatched robustness protocol.
- No P1-5 disease-classification result is used.
