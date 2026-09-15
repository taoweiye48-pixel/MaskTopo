# TopoBridge post-processing optimization protocol

## Purpose

Reduce the two remaining CPU stages identified by
`TB-B-260731-004`: patch component majority/limit and fixed-K assignment.  The
new paths must reproduce the existing artifact semantics before their speed is
considered.

## Frozen setup

- FIVES held-out mask probabilities from `results_fives_seed20260810`.
- Dev-selected mask threshold 0.2 and zero closing iterations.
- Patch grid 16 x 16 and reduced token budget K=8.
- Batches 1, 8, and 32; 20 warm-up and 100 measured repetitions.
- CPU wall-clock timing for this post-processing gate.  Dataset I/O and mask
  predictor inference are excluded because they are unchanged.

## Compared paths

1. Reference patch grouping/limit: the existing per-patch `np.unique` and
   Python remapping implementation.
2. Vectorized patch grouping/limit: a batched 4 x 4 block mode calculation with
   deterministic smallest-label tie breaking and vectorized component pruning.
3. Reference assignment: the existing `oracle_assignment` implementation.
4. Vectorized assignment: the same count allocation and farthest-seed policy,
   with array-based multi-source shortest-path relaxation and deterministic
   tie breaking.
5. Full reference post-processing: reference groups + reference assignment +
   per-sample graph construction.
6. Hybrid optimized post-processing: vectorized groups + the exact reference
   assignment + batched graph/reachability construction.
7. Fully vectorized candidate: vectorized groups + vectorized assignment +
   batched graph/reachability construction.  This is retained as a negative or
   positive implementation control; it is not selected merely for being more
   vectorized.

## Correctness gate

Before timing, require exact equality on at least 128 real held-out samples and
additional synthetic patch-group patterns:

- vectorized patch groups equal reference groups;
- vectorized assignments equal reference assignments;
- adjacency and reachability equal after both paths;
- a random-initialized standard ViT output is allclose (`atol=1e-6`,
  `rtol=1e-5`) for equivalent artifacts.

Any mismatch stops the run and marks the candidate non-equivalent.

## Interpretation

This gate supports an implementation decision, not a new accuracy claim.  A
candidate may be plugged into the end-to-end benchmark only if its exactness
gate passes.  The online efficiency claim remains limited by the prior
batch=1/8/32 end-to-end protocol.
