---
date: 2026-08-02
experiment_id: TB-B-260802-023
title: Unified fixed-K PH-inclusive Pareto
status: frozen_before_new_k_runs
route: route_2_topology_preservation_efficiency
---

# Unified fixed-K PH-inclusive Pareto protocol

## Scope

- Datasets: DeepCrack and CrackForest.
- Token budgets: K=8, 16, 32, 64.
- Optimization seeds: 20260810, 20260811, 20260812.
- PH models: PH-only and PH-guided at every K; neither may be dropped because of the K=16 outcome.
- Existing comparators: MaskTopo, Grid, TokenLearner, Perceiver Resampler, ToMe-style, and mask-guided queries from the frozen token-budget runs.
- FIVES remains a K=8 direct-collision result because a complete matched MaskTopo/reducer K curve was not frozen there.
- All test sets were previously observed. This stage is retrospective Pareto evidence, not new external confirmation.
- Massachusetts Roads test remains sealed.

## Frozen training contract

- Reuse the seed-specific predicted-mask probability and PH descriptor cache from TB-B-260802-022.
- Same arrays, PatchStem, common head, dimension 64, 22 epochs, batch size 32, AdamW, learning rate 1e-3, weight decay 1e-4, augmentation, dev-only checkpoint selection, and test threshold.
- K=16 results are reused byte-for-byte; they are not retrained.
- New outputs are written to distinct K/seed directories; failures and retries are never overwritten.

## Pareto axes

- Accuracy: three-seed balanced accuracy and all seed values.
- Structural outcome: endpoint-connectivity balanced accuracy; existing MaskTopo direct-structure diagnostics are retained separately.
- Cost: parameter count, FLOPs where supported by the frozen profiler, peak CUDA memory, PH/component preprocessing time, and conservative end-to-end latency at batch 1/8/32.
- PH preprocessing is never hidden as offline-free work. The primary PH latency adds measured cubical-PH CPU time to GPU model time.
- Existing baseline latency values measured under incomplete batch conditions may not be mixed into the unified table; models lacking batch 1/8/32 measurements must be re-benchmarked from frozen checkpoints or marked missing.

## Reporting contract

- Plot/report all K values, including CrackForest K=64 and batch=1 failures.
- Report PH-only and PH-guided separately plus a clearly labeled best-PH envelope; the envelope is descriptive and may not be called pre-specified model selection.
- Do not compare TopoCL or the WACV 2026 classifier numerically because their tasks and compute protocols differ.
- Do not use the P1-5 disease-classification +1 pp result anywhere in this Pareto claim.
