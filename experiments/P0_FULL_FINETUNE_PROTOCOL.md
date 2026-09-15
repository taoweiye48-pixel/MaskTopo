# TopoBridge P0 full-fine-tuning and counterfactual protocol

## Scope

This protocol completes the three P0 gates identified after the trained ViT
pilot: a fully fine-tuned fairness comparison, topology-mechanism ablations,
and shuffled/random-topology counterfactual controls.

## Frozen data and backbone

- FIVES matched crops v2: train/dev/test = 2400/600/1200.
- Official torchvision ViT-B/16 ImageNet-1K V1 initialization.
- Input 64 x 64 -> bilinear 256 x 256, ImageNet normalization.
- Reduction after transformer block 2, K=8.
- All ViT parameters are trainable except the unused original 1000-class head.
- Micro-batch 8, gradient accumulation 1, effective batch 8. A pre-run gate
  measured about 1.9 GB peak allocated CUDA memory, so accumulation is not
  needed on the RTX 3060. The implementation retains accumulation as a fallback.
- AdamW: backbone LR 1e-5, new-layer LR 1e-3, weight decay 0.05.
- Eight epochs, cosine schedule with 5% warm-up, AMP, gradient clipping 1.0.
- Radiometric augmentation only; no geometric transform can break cached
  topology alignment.
- Seeds 20260810, 20260811, 20260812.
- Checkpoint selection uses development balanced accuracy only. Test is evaluated
  exactly once after loading the best development checkpoint.

## Conditions

### Fully fine-tuned comparison

- `full`: no token compression.
- `grid`: rectangular K=8 pooling.
- `g2tm_fixedk`: feature-similarity maximum-spanning-forest K=8 pooling.
- `hybrid`: predicted-mask assignment plus local adjacency and reachability.

### Mechanism ablations

- `assignment_only`: predicted-mask assignment, no graph message.
- `local_only`: assignment plus local adjacency message only.
- `reachability_only`: assignment plus reachability message only.
- `hybrid`: both messages.

Local-only, reachability-only and Hybrid instantiate the same two message
projections. Disabled branches remain present but make no forward contribution,
so differences are not explained by adding a different classifier head.

### Counterfactual controls

- `shuffled_global`: use another held-out sample's complete artifact tuple via a
  deterministic derangement.
- `shuffled_within_class`: derange complete artifacts only among samples with
  the same class. This diagnoses label-level topology information separately
  from image-artifact alignment.
- `random_connected`: deterministic random connected spatial Voronoi partitions
  with a coarse graph induced on the full background grid.

All three controls retain K=8 and the same Hybrid graph architecture.

## Pre-registered P0 gates

1. Full-fine-tuning gate: Hybrid >= Full - 2 percentage points and Hybrid >=
   G2TM + 3 points in mean held-out balanced accuracy.
2. Counterfactual gate: Hybrid exceeds both global-shuffled and random-connected
   controls by at least 5 points.
3. Mechanism findings are diagnostic: report Hybrid minus assignment-only,
   local-only and reachability-only. A component that adds <1 point is not a
   defensible independent contribution.
4. Within-class shuffle is diagnostic, not a hard gate. If its drop is <3
   points, disclose that the topology branch may chiefly carry label-level
   structure rather than precise image-artifact alignment.

No p-value is claimed from three seeds. Report mean, sample SD, paired seed
differences and the raw held-out predictions for later source-cluster bootstrap.

## Stop rules

- Stop a run for NaN/Inf, artifact misalignment, failed derangement, checkpoint
  reload mismatch, CUDA OOM, or missing test-source identity.
- A smoke/timing run is marked non-paper and stored separately.
- Do not silently retry a failed formal run; record the failure before fixing or
  restarting.
