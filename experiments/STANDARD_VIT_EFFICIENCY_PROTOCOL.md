# Standard ViT-B/16 token efficiency protocol

This is the first standard-backbone efficiency gate. It deliberately separates
compute evidence from accuracy evidence.

## Frozen design

- Backbone: torchvision ViT-B/16 with 256x256 input, 16x16 patch grid.
- Insertion point: after transformer layer 2.
- Fine token count: 256 patch tokens plus CLS.
- Reduced token budget: K=8.
- Modes: full tokens, rectangular grid pooling, G2TM-FixedK feature grouping,
  and cached MaskTopo assignment plus one graph message layer.
- Batch: 8.
- Warmup/repeats: 10/30.
- Device: local CUDA GPU.

## Interpretation

This run measures end-to-end transformer forward latency and peak CUDA memory
for a standard ViT architecture. The backbone is randomly initialized, so the
run must not be used to claim accuracy or task performance. The next accuracy
gate must train the same standard backbone on the endpoint-connectivity task,
with identical optimization and supervision across modes.

The cached MaskTopo mode excludes mask-predictor and CPU artifact-construction
time. Those costs are reported separately in the existing TopoBridge reports
and must be included in the final end-to-end paper table.
