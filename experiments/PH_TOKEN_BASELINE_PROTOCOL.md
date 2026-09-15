---
date: 2026-08-02
experiment_id: TB-B-260802-022
title: PH-based fixed-K token direct topology baseline
status: frozen_before_implementation
route: route_2_topology_preservation_efficiency
verification_status: protocol_frozen
---

# PH-based fixed-K token direct topology baseline

## Material Passport

- Origin: reviewer-risk audit after TB-B-260802-021
- Purpose: directly compare component-aligned MaskTopo with persistent-homology-based token representations
- Confirmatory status: retrospective collision on previously observed FIVES, DeepCrack, and CrackForest tests
- Excluded confirmatory set: Massachusetts Roads one-shot test remains sealed; no PH result may be added to its frozen verdict
- Primary task: endpoint-connectivity classification, not FIVES disease classification P1-5

## Frozen research question

Under the same predicted-mask input, exact token budget, patch encoder, downstream head, training budget, and seed, does an explicit spatial connected-component representation retain endpoint connectivity better than a persistent-homology descriptor representation?

## Frozen datasets and primary budgets

| Dataset | Primary K | Seeds | Status |
|---|---:|---|---|
| FIVES | 8 | 20260810, 20260811, 20260812 | retrospective collision |
| DeepCrack | 16 | 20260810, 20260811, 20260812 | retrospective collision |
| CrackForest | 16 | 20260810, 20260811, 20260812 | retrospective mechanism/collision |

After the primary-budget runs finish, the surviving PH implementation enters the unified K=8/16/32/64 Pareto experiment. No model or hyperparameter may be selected using test results.

## Frozen PH construction

1. Input is the seed-matched predicted foreground probability map used by MaskTopo. Ground-truth masks/topology are forbidden at inference.
2. Use a two-dimensional cubical complex with top-cell filtration `f = 1 - clip(mask_probability, 0, 1)`, i.e. a foreground-probability superlevel filtration.
3. Compute persistence in homology dimensions H0 and H1.
4. Encode each persistence pair as filtration birth, filtration death, persistence, homology type, essential flag, validity flag, and normalized birth/death critical-cell coordinates.
5. Finite and essential pairs compete in one deterministic descending-persistence ranking. Ties are broken by homology dimension and critical-cell indices. Select exactly K pairs; if fewer than K exist, append zero descriptors with validity flag 0. The learned model receives no ground-truth-derived padding information.
6. Report the selected H0/H1/essential counts and persistence distribution. Do not describe the persistence diagram as spatially equivalent to a component assignment: it is a global topological summary with only critical-cell anchors.

## Frozen models

### PH-only

- Map the K PH descriptors through a learned MLP to K tokens.
- Use descriptor birth coordinates and normalized persistence as the three metadata channels expected by the common head.
- Use identity adjacency/reachability because a persistence diagram does not define the MaskTopo component graph.
- This is the diagnostic test of whether a global PH summary alone is sufficient.

### PH-guided

- Use the same PatchStem that produces 16x16 visual context tokens.
- Map the K PH descriptors to K queries.
- Apply two cross-attention/feed-forward resampler blocks over the visual context, matching the existing Perceiver-style depth.
- Produce exactly K visual-topological tokens and use the same common classifier head with identity adjacency/reachability.
- This is the primary direct topology baseline.

## Fairness controls

- Same RGB/marker input and same seed-matched predicted mask probability as the input-matched baselines.
- Same exact K, embedding dimension, optimizer, learning rate, weight decay, epoch count, batch size, data augmentation, dev checkpoint rule, and test threshold.
- Same train/dev/test arrays and source split fingerprints as prior frozen runs.
- Test is evaluated only after selecting the best checkpoint by dev balanced accuracy.
- Save command, dependency versions, code hashes, mask hash, checkpoint hash, predictions, probabilities, history, PH cache hash, and all failures.

## Metrics

- Primary: test balanced accuracy.
- Secondary: accuracy, sensitivity, specificity, dev balanced accuracy, parameter count, training time.
- Topology diagnostics: selected H0/H1 counts, essential-pair count, persistence quantiles, descriptor padding rate.
- Efficiency: PH preprocessing latency, GPU model latency, conservative summed end-to-end latency, peak CUDA memory at batch 1/8/32; FLOPs are added in the unified Pareto stage.
- Statistics: mean and standard deviation over three seeds plus paired source-cluster bootstrap difference against MaskTopo and the strongest input-matched reducer. CrackForest remains diagnostic because its test was used during development.

## Interpretation contract

- If MaskTopo exceeds PH-guided with a positive paired confidence interval, this supports the narrower claim that spatial component membership and explicit graph messages provide value beyond global PH descriptors for fixed-budget endpoint reasoning.
- If PH-guided ties or wins, topology-superiority claims must be downgraded; the contribution may remain an efficiency, spatial interpretability, or inductive-bias result.
- A weak PH-only result cannot by itself establish MaskTopo superiority; the primary comparison is PH-guided.
- TopoCL and the WACV 2026 multi-scale/multi-filtration method are protocol-related work, not numerical head-to-head baselines, because they study medical-image classification rather than exact-K endpoint-connectivity coarsening.

## Stop and contamination rules

- Do not run, inspect, or alter Massachusetts Roads test predictions, test masks, or one-shot verdict.
- A code failure is logged and diagnosed; it is not silently retried under the same experiment ID.
- Any implementation change after a failed run receives a retry suffix and an anomaly record.
- All artifacts and logs are written under `E:\TopoBridge_MVP` and `E:\TopoBridge_MVP\实验记录`.
