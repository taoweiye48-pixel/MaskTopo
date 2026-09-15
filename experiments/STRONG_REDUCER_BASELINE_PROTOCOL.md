# TopoBridge strong token-reducer baseline protocol

## Frozen question

Under the same image stem, 16-token budget, downstream classifier, data
splits, optimization budget, and random seeds, does MaskTopo outperform
strong learned or similarity-based token reducers?

## Frozen datasets and seeds

- CrackForest: cached 1,600 / 320 / 320 samples.
- DeepCrack matched-v2: cached 2,400 / 600 / 1,200 samples.
- Optimization seeds: 20260810, 20260811, 20260812.
- Image size: 64 x 64.
- Fine tokens: 16 x 16 = 256.
- Reduced tokens: 16.
- Epochs: 22.
- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4.
- Checkpoint selection: development balanced accuracy only.
- Classifier threshold: 0.5, not calibrated on test.

The held-out labels were already observed in earlier mechanism development.
This comparison is therefore a frozen retrospective benchmark, not a new
one-shot confirmation.

## Reducers

1. `tokenlearner`
   - Spatially conditioned learned attention maps.
   - Each map pools all 256 fine tokens into one reduced token.

2. `perceiver_resampler`
   - Sixteen learned latent queries.
   - Two cross-attention + feed-forward blocks over position-aware fine tokens.

3. `tome_style`
   - Four rounds of content-similarity bipartite soft merging:
     256 -> 128 -> 64 -> 32 -> 16.
   - This is an in-framework differentiable ToMe-style control, not the
     authors' official ToMe implementation.

4. `mask_guided_queries`
   - The same resampler architecture as `perceiver_resampler`.
   - Receives the predicted mask probability as an additional embedding.
   - Uses no connected components, hard assignment, adjacency, or graph
     message passing.

All reducers feed the same projection and coarse-token Transformer head.
The baseline head receives identity adjacency and identity reachability.

## Fairness constraints

- No ground-truth mask or topology is available at inference.
- Only `mask_guided_queries` receives the already-frozen predicted mask,
  matching the information available to MaskTopo.
- No test metric may choose a reducer hyperparameter or checkpoint.
- All runs use identical augmentation, batch size, epoch count, optimizer,
  development selection rule, and classification threshold.
- Every result records parameter count, training time, batched latency, and
  peak CUDA memory where available.

## Frozen gate

The strong-baseline gate passes only if:

1. MaskTopo exceeds the strongest baseline by at least 2 percentage points on
   both datasets in the three-seed mean;
2. all three per-seed MaskTopo-minus-baseline differences are positive on both
   datasets;
3. the source-image cluster bootstrap 95% confidence interval on DeepCrack has
   a lower bound above zero.

If the gate fails, the defensible claim is limited to a topology-specific
inductive bias rather than a generally superior token reducer.

