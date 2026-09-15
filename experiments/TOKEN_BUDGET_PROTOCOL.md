# TopoBridge token-budget and efficiency protocol

## Frozen research question

Does MaskTopo retain an accuracy advantage over learned and
similarity-based reducers as the visual-token budget changes, especially in
the low-token regime?

## Frozen budgets, models, and runs

- Reduced token counts: `K = 8, 16, 32, 64`.
- Fine image tokens: 256.
- Models:
  - MaskTopo;
  - TokenLearner;
  - Perceiver Resampler;
  - differentiable ToMe-style merging;
  - mask-guided learned queries without explicit topology.
- Datasets: CrackForest and shortcut-matched DeepCrack.
- Optimization seeds: 20260810, 20260811, 20260812.
- Training configuration: the same 22 epochs, AdamW, learning rate,
  augmentation, batch size, development checkpoint selection, and fixed 0.5
  test threshold as the frozen `K=16` experiment.
- `K=16` accuracy results are reused without retraining.
- `K=8, 32, 64` are new runs.

All held-out labels were observed before this stage. This is a frozen
retrospective budget analysis, not an untouched one-shot confirmation.

## Variable-token MaskTopo construction

The predicted mask threshold and morphological closing setting remain those
selected on each prior development run. Connected components are limited to
at most `K-1` foreground groups, reserving a background group. All 256 fine
patches are assigned to exactly `K` non-empty coarse tokens. No test setting
changes thresholding, group construction, checkpoint selection, or the
classifier decision threshold.

## Accuracy outputs

For every dataset, model, and budget:

- three per-seed balanced accuracies;
- mean and sample standard deviation;
- paired MaskTopo-minus-reducer gains;
- source-image cluster bootstrap confidence intervals;
- balanced-accuracy versus `log2(K)` area under the curve.

The per-budget baseline envelope is the highest three-seed mean among the
four reducer baselines at that budget. It is intentionally stricter than
choosing one fixed competitor.

## Efficiency outputs

Report separately:

1. reducer/classifier parameters;
2. reducer/classifier batched latency and peak CUDA memory;
3. PyTorch-profiler-counted forward FLOPs, explicitly labeled as a partial
   operator count rather than a hardware-independent complete FLOP audit;
4. predicted-mask inference cost for MaskTopo and mask-guided queries;
5. CPU connected-component and graph-construction time for MaskTopo.

Both connector-only and end-to-end estimates must remain visible. Mask
prediction or graph construction may not be silently excluded from the
end-to-end comparison.

## Frozen continuation gates

The stage passes only if all conditions hold:

1. at `K=8`, MaskTopo exceeds the per-budget strongest baseline by at least
   2 percentage points on both datasets;
2. all three paired `K=8` gains are positive on both datasets;
3. the DeepCrack source-cluster bootstrap 95% CI lower bound for the `K=8`
   strongest-baseline contrast is above zero;
4. on both datasets, MaskTopo's log-budget AUC exceeds the AUC of the
   per-budget strongest-baseline envelope by at least 2 percentage points;
5. MaskTopo `K=8` mean balanced accuracy is no more than 5 percentage points
   below its `K=16` mean on each dataset.

Efficiency is descriptive in this stage. A slower end-to-end path does not
invalidate the mechanism, but it must be treated as a deployment limitation
and motivates later reliability-aware conditional routing.

