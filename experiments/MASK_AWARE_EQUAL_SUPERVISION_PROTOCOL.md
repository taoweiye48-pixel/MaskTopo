---
experiment_id: TB-B-260803-032
title: DeepCrack K16 equal-supervision mask-aware non-topological baselines
status: frozen_before_test_run
dataset: DeepCrack
task: endpoint_connectivity
token_count: 16
optimization_seeds: [20260810, 20260811, 20260812]
split_seed: 20260730
primary_statistical_unit: source_image
confirmatory_status: retrospective_collision
---

# DeepCrack K=16 equal-supervision mask-aware non-topological protocol

## Question

Does MaskTopo retain an endpoint-connectivity advantage when the comparator receives the same dense-mask supervision product—the identical saved predicted-mask probability—but is forbidden to construct connected components, adjacency graphs, reachability, persistence homology, or any other topology representation?

This experiment tests topology-specific organization versus mask-conditioned token reduction. It does not test disease classification, and P1--P5 evidence is excluded.

## Frozen data and training

- DeepCrack frozen train/dev/test arrays: 2,400/600/1,200 crops; split seed 20260730.
- The crop/data seed is fixed to 20260810 for every optimization seed. Only the optimization seed and its paired saved float16 mask archive vary across runs.
- Test statistical unit: source image; 1,200 crops map to 222 source images.
- Each optimization seed uses the same saved train/dev/test mask-probability archive as the aligned MaskTopo and PH experiments. All archive arrays must be stored as IEEE float16 before loading.
- K=16 exactly; common token dimension 64.
- 22 epochs, batch 32, AdamW, learning rate `1e-3`, weight decay `1e-4`, identical image augmentation, optimization seeds 20260810/11/12, and development-only checkpoint selection.
- The unchanged `CommonTokenClassifier / CoarseGraphHead` is used with identity adjacency and identity reachability, so the common head receives no cross-token topology graph.
- DeepCrack test was previously observed. Each development-selected seed is evaluated once; this is a retrospective collision, not untouched confirmation.

## Frozen baselines

### 1. `mask_conditioned_perceiver_strong`

- Inputs: PatchStem image features, fixed patch coordinates, and the identical float16-stored predicted-mask probability downsampled to the 16x16 patch grid.
- Exact K: 16 learned latent queries.
- Mask conditioning: mask-value embedding is added to every context token, and each cross-attention block receives a learned per-query foreground logit bias derived from the mask probability.
- Two Perceiver cross-attention/feed-forward blocks.
- No connected components, component labels, graph adjacency, reachability, skeleton, PH, or topology loss.
- Capacity deliberately exceeds MaskTopo; this is the strongest non-topological challenge, not a parameter-favorable comparison for MaskTopo.

### 2. `mask_conditioned_slot_param_matched`

- Inputs: the same PatchStem image features, fixed patch coordinates, and identical mask probability.
- Exact K: 16 learned slots.
- Two rounds of normalized dot-product slot-to-patch attention with a learned per-slot foreground logit bias; weighted visual pooling produces the K tokens.
- No learned Q/K/V or feed-forward resampler blocks. This keeps total capacity within ±1% of MaskTopo while retaining direct mask conditioning.
- No connected components, component labels, graph adjacency, reachability, skeleton, PH, or topology loss.

## Capacity rule

- Frozen MaskTopo parameter count: 149,123.
- Existing `mask_guided_queries`: 259,011 parameters, already larger than MaskTopo.
- Parameter-matched baseline target: within ±1% of 149,123 without altering the common head or token dimension.
- The full baseline is not reduced to match capacity; both the full and matched results must be reported.

## Comparators

- Float16-aligned MaskTopo predictions from TB-B-260802-026.
- Existing `mask_guided_queries` predictions from the frozen strong-reducer runs.
- `mask_conditioned_perceiver_strong` and `mask_conditioned_slot_param_matched` from TB032.

No test-based model selection or post-hoc best-baseline substitution is allowed.

## Statistical analysis and claim gate

- Primary: aligned MaskTopo minus `mask_conditioned_perceiver_strong`.
- Capacity-isolated secondary: aligned MaskTopo minus `mask_conditioned_slot_param_matched`.
- Existing-baseline sanity: aligned MaskTopo minus frozen `mask_guided_queries`.
- Paired source-image cluster bootstrap: 20,000 repetitions; statistics seed 20260803; report seed-wise values, mean differences, and percentile 95% confidence intervals.
- The topology-specific DeepCrack claim passes this final fairness gate only if both new MaskTopo-minus-baseline confidence-interval lower bounds are greater than zero.
- If either interval crosses zero, the claim is narrowed to effective mask-conditioned token reduction; no topology-specific superiority claim is made from DeepCrack.
- If either baseline has a confidence interval wholly above MaskTopo, the DeepCrack topology-specific superiority claim is withdrawn.

## Execution gates

1. Verify CUDA, no active TB032 process, float16 archive storage, frozen sample counts, and the MaskTopo parameter anchor.
2. Architecture self-test must assert exact `[batch,16,64]` tokens, active gradients, identity-only classifier graph inputs, and absence of prohibited topology operations/imports.
3. The matched model parameter error must be within ±1% before any test evaluation.
4. Run seeds sequentially. A crash is reported and is not silently retried.
5. Validate stored predictions against recorded metrics and frozen truth/source IDs before bootstrap.

## Required records

- All runtime stdout/stderr and anomaly records under `E:\TopoBridge_MVP\实验记录`.
- Per-seed checkpoints, histories, predictions, and `result.json` under `results_mask_aware_equal_supervision_deepcrack_k16_seed*`.
- Final report, source-level statistical table, artifact hashes, and evidence-matrix update under the project and experiment-record directories.
