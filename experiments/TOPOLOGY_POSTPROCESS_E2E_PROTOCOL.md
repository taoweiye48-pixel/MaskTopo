# TopoBridge optimized post-processing end-to-end protocol

## Purpose

Confirm whether the exact hybrid selected by
`TOPOLOGY_POSTPROCESS_OPTIMIZATION_PROTOCOL.md` improves actual end-to-end
MaskTopo latency, including mask prediction, CPU construction, artifact
transfer, token pooling, graph mixing, and the standard ViT-B/16 forward.

## Frozen setup

- FIVES held-out cached crops and frozen mask checkpoint.
- Mask threshold 0.2, closing iterations 0.
- torchvision ViT-B/16, input 256 x 256, compression after layer 2, K=8.
- Batches 1, 8, and 32; 20 warm-up and 100 synchronized measurements.
- Compared paths: Full ViT, original CPU reference MaskTopo, optimized hybrid
  MaskTopo, precomputed-artifact upper bound, and mask-only/grid ablation.

The hybrid uses vectorized patch grouping/limit, the unchanged reference
fixed-K assignment, and batched graph/reachability construction.

## Correctness gate

On 128 held-out masks, require exact assignment, adjacency, and reachability
versus the reference path.  On eight images, require downstream ViT outputs to
be allclose with `atol=1e-6, rtol=1e-5`.

## Interpretation gates

- at least 30% mean end-to-end reduction versus Full ViT: solid supporting
  efficiency result;
- 20% to less than 30%: report, but not as the core contribution;
- less than 20%: do not make efficiency central for that batch regime;
- precomputed artifacts are an offline upper bound and never count as online
  inference.

Random initialization is used for latency only and cannot support an accuracy
claim.
