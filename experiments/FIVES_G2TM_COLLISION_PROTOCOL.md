# FIVES G2TM collision protocol

Status: retrospective collision test. The FIVES held-out results had already
been observed before this protocol, so this experiment is diagnostic rather
than a new confirmatory test.

## Question

Does MaskTopo still outperform the closest feature-graph connected-component
token reducer when both methods use the same FIVES samples, PatchStem,
downstream head, optimizer, epoch budget, optimization seeds, and exact K=8
token budget?

## Compared models

- `g2tm_fixedk_feature`: four-neighbour feature cosine graph, with a maximum
  spanning forest stopped at exactly eight connected components.
- `g2tm_fixedk_mask`: the same reducer, with the same predicted mask
  probabilities available to MaskTopo embedded into the grouping features.
- Existing frozen `mask_topo_external` results are used as the target method.

The collision baselines preserve the official G2TM core idea but are a matched
fixed-budget adaptation, not an official reproduction of the paper's Segmenter
results. The audited official source is:

- Repository: https://github.com/vbercy/g2tm-segmenter
- Branch: `torch2`
- Revision: `f17d1c8374a6f09365d856ff40d5aaf6d0bcf5d4`

## Frozen settings

- Dataset and samples: exactly the prior FIVES split and cached crops.
- Fine tokens: 16x16 = 256.
- Reduced tokens: K=8.
- Epochs: 22.
- Batch size: 32.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001.
- Seeds: 20260810, 20260811, 20260812.
- Checkpoint selection: development balanced accuracy only.
- Held-out threshold: fixed at 0.5.
- Primary metric: held-out balanced accuracy.
- Uncertainty: source-image-cluster bootstrap over paired predictions.

## Decision gate

- Continue the current method claim if MaskTopo beats the strongest matched
  G2TM variant by at least 3 percentage points and the source-cluster bootstrap
  95% interval has lower bound above zero.
- Treat current connected-component tokenization novelty as weak if the gap is
  at most 1-2 percentage points and G2TM is faster.
- Pivot the method contribution toward quotient-graph reachability and
  topology-critical anchors if a matched G2TM variant ties or wins.

## Leakage and interpretation

No ground-truth test mask is provided at inference. The mask-guided collision
variant receives only the saved predicted mask probabilities from the
corresponding prior optimization seed. Because the FIVES test was already
observed, all conclusions remain retrospective and require a later untouched
task or dataset for confirmation.
