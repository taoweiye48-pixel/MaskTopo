---
date: 2026-08-02
experiment_id: TB-B-260802-023E
parent_experiment: TB-B-260802-023
title: Frozen-checkpoint same-session efficiency addendum
status: frozen_before_benchmark
---

# PH-inclusive Pareto efficiency addendum

## Frozen benchmark scope

- Hardware/session: the currently available NVIDIA GeForce RTX 3060; one uninterrupted benchmark process.
- Connector weights: optimization seed 20260810 frozen checkpoints; no retraining and no checkpoint selection.
- Connector input: frozen DeepCrack test arrays and predicted-mask probabilities, first 1/8/32 samples in fixed order.
- Models at K=8/16/32/64: MaskTopo, TokenLearner, Perceiver Resampler, ToMe-style, mask-guided queries, PH-only, and PH-guided.
- Warm-up/repetitions: 10/30 for every GPU connector and mask-predictor batch measurement.
- CUDA peak memory includes the loaded connector, its exact-size input batch, and forward intermediates.
- FLOPs: batch-1 CPU `torch.profiler(with_flops=True)` sum, explicitly a partial supported-operator count rather than complete theoretical FLOPs.
- Before timing, each loaded checkpoint must reproduce its saved seed-20260810 held-out binary predictions.

## Preprocessing and end-to-end accounting

- The frozen DeepCrack seed-20260810 mask predictor is benchmarked at batch 1/8/32 and verified against cached mask probabilities within float16 storage tolerance.
- CPU MaskTopo component/assignment/graph construction and cubical-PH descriptor extraction are measured for batch 1/8/32 on both DeepCrack and CrackForest frozen seed-20260810 probabilities.
- Conservative serial latency per sample:
  - TokenLearner / Perceiver / ToMe: connector only.
  - mask-guided queries: mask predictor + connector.
  - MaskTopo: mask predictor + component/graph preprocessing + connector.
  - PH-only / PH-guided: mask predictor + cubical-PH preprocessing + connector.
- Serial pipeline peak CUDA memory is the maximum of the mask-predictor and connector peaks; CPU preprocessing memory is not represented as CUDA memory.
- Pipeline parameters and profiler-counted GPU FLOPs include the mask predictor only for methods that require predicted masks. CPU topology/PH operations are marked unsupported for FLOP counting.

## Reporting boundaries

- Connector timing is a same-architecture measurement and is shared across the two dataset accuracy rows; dataset-specific CPU preprocessing remains separate.
- Accuracy is read only from the already verified three-seed Pareto summary.
- All test outcomes are retrospective; Massachusetts Roads remains sealed.
- No P1-5 disease-classification result is used.
