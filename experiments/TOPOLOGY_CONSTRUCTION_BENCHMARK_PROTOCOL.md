# TopoBridge topology-construction efficiency protocol

## Purpose

Measure and optimize the online overhead that converts a predicted dense mask
into the fixed-budget TopoBridge artifacts.  This experiment is an efficiency
gate only; it does not change or re-evaluate task accuracy.

## Frozen setup

- Dataset: held-out FIVES test crops from the frozen split.
- Mask checkpoint: `results_fives_seed20260810`.
- Mask threshold and closing iterations: read from the frozen dev-selected
  configuration.  The expected values are threshold 0.2 and zero closing
  iterations.
- Backbone: torchvision ViT-B/16, 256 x 256 input.
- Compression point: after transformer layer 2.
- Fine/reduced patch-token counts: 256 / K=8.
- Batch sizes: 1, 8, and 32.
- Timing: 20 warm-up iterations and 100 measured iterations.
- Statistics: mean, median, standard deviation, and P95 wall-clock latency,
  reported per batch and per sample.
- Device: the local CUDA GPU for the mask predictor and ViT; CPU and GPU paths
  are both tested for topology construction.

## Compared construction paths

1. `cpu_reference`: the current per-sample SciPy connected components,
   per-sample oracle assignment, and per-sample graph/reachability code.
2. `cpu_batch`: the same connected components and assignment semantics, with
   graph and transitive-closure construction vectorized across the batch.
3. `gpu_hybrid`: exact GPU label propagation for connected components followed
   by the unchanged CPU assignment and batched graph construction.
4. `precomputed`: cached assignment/graph tensors.  This is an offline upper
   bound and must not be presented as ordinary online inference.
5. `mask_only`: mask predictor plus fixed-grid token compression, without
   connected components, topology assignment, or graph construction.  This is
   an efficiency ablation and requires a separate trained-accuracy experiment.

## Required correctness gate

Timing results are interpretable only if all of the following pass on at least
128 held-out samples:

- thresholded binary masks are identical;
- connected-component partitions are equivalent after canonical relabeling;
- 16 x 16 patch groups are identical;
- assignments, adjacency, and reachability are exactly equal;
- downstream ViT outputs from equivalent artifacts are numerically equal
  within `atol=1e-6, rtol=1e-5`.

Any failing path is reported as non-equivalent and excluded from the speed
comparison.

## Timing decomposition

- mask predictor;
- thresholding, connected components, and patch grouping;
- fixed-K assignment;
- graph plus reachability construction;
- CPU-to-GPU artifact transfer;
- token pooling;
- compressed ViT forward;
- measured end-to-end inference.

GPU mask probabilities remain on the GPU for the GPU path.  The CPU paths
include the required GPU-to-CPU probability transfer in end-to-end timing.
Process start-up, dataset loading, checkpoint loading, and one-time cache
creation are excluded.

## Pre-registered interpretation gates

- at least 30% end-to-end latency reduction versus Full ViT: efficiency may be
  treated as a solid supporting contribution;
- 20% to less than 30%: efficiency may be reported, but should not be the core
  contribution;
- less than 20%: do not make efficiency a central claim; frame the method as a
  topology-preserving visual connector;
- if the GPU connected-component path does not beat the best CPU online path,
  retain the faster CPU implementation and report the negative GPU result.

`precomputed` is reported separately and is never used to decide the online
efficiency claim.
