# FIVES third-domain external confirmation protocol

Frozen on: 2026-07-31, before any FIVES downstream model test performance was
observed.

## Research question

Does topology-supervised token coarsening retain a reproducible advantage over
strong in-framework token reducers on a non-crack thin-structure domain:
retinal-vessel connectivity reasoning?

## Dataset and split

- Dataset: FIVES, Figshare record `10.6084/m9.figshare.19688169.v1`.
- License recorded by Figshare: CC BY 4.0.
- Official layout: 600 train images and 200 held-out test images, each with a
  2048 x 2048 color fundus image and a pixelwise vessel mask.
- The official 200-image test set remains held out for model selection.
- The official 600-image train set is deterministically stratified by disease
  suffix into 480 train and 120 development source images.
- Split seed: `20260731`.
- Subject identifiers are not available in the released filenames. Therefore,
  patient-level overlap cannot be independently audited and remains a declared
  limitation.

## Derived task

The 2048 x 2048 images are first converted to a green-channel 512 x 512
working image. The shared sample generator then performs its standard
single inversion so vessels have a bright response, matching the two earlier
domains. Vessel masks use 4 x 4 max pooling so thin annotated vessels are not
deleted by downsampling. For each 64 x 64 working crop
(equivalent to a 256 x 256 source crop), two visible vessel pixels are marked.
Predict whether the two pixels belong to the same connected component in the
manual vessel annotation.

- Crop input is 64 x 64 after the fixed preprocessing above.
- Manual masks are used only to generate labels and train the global mask
  predictor.
- Held-out inference receives the fundus crop and two endpoint markers, never
  the manual mask or manual topology.
- Positive and negative examples are exactly balanced.
- Candidate examples are matched within mean-intensity and endpoint-distance
  bins before the requested sample count is selected.
- Derived sample counts: 2400 train, 600 development, 1200 held out.
- Statistical unit for confidence intervals: original fundus image.

## Models

All models use the same PatchStem, downstream classifier, optimizer,
development-only checkpoint selection and K=8 reduced-token budget.

Primary method:

- MaskTopo: predicted vessel components -> component-aligned token pooling ->
  predicted-topology message passing.

Strong reducer controls:

- TokenLearner;
- two-layer Perceiver Resampler;
- differentiable ToMe-style similarity merging;
- mask-guided queries without connected components or topology graph.

Mechanism controls:

- fixed grid;
- predicted mask as an additional image feature;
- component assignment without graph message passing;
- grid pooling with predicted mask graph;
- spatially shuffled topology.

The ToMe-style and other reducer controls are in-framework implementations, not
claims about the authors' official repositories.

## Optimization

- Seeds: `20260810`, `20260811`, `20260812`.
- Mask predictor epochs: 30.
- Downstream epochs: 22.
- Batch size: 32.
- AdamW learning rate: 1e-3.
- Weight decay: 1e-4.
- Classification threshold: fixed at 0.5.
- Mask threshold and morphological closing are selected on development data
  only.

## Data gate

Training may start only if all conditions pass:

1. train/dev/test source-image intersections are empty;
2. every split is exactly class balanced;
3. held-out endpoint-distance shortcut accuracy is at most 55%;
4. held-out mean-intensity shortcut accuracy is at most 55%;
5. at least 180 of the 200 official test images contribute examples;
6. all derived arrays and manual-mask reconstructions pass shape and label
   invariants.

If a data gate fails, no FIVES model test performance may be inspected. A
generic sampling correction may be frozen and the data gate rerun, with every
version retained.

Data-audit revision record:

- v1 stopped before training because it required the centroid of every circular
  endpoint marker to land on one exact vessel pixel. Markers clipped by a crop
  boundary can have a shifted centroid even though the marked endpoint and its
  patch remain valid.
- Before any FIVES model performance was observed, v2 replaced that
  representation-sensitive check with two label-preserving invariants:
  each marker channel must overlap the reconstructed manual vessel mask, and
  the two endpoint patches' manual component IDs must agree exactly with the
  connected/disconnected label. All v1 artifacts remain retained.
- Visual inspection of v2 derived examples then identified a double inversion:
  the preparation step and shared crop generator both inverted intensity.
  Before training, v3 froze a new `preprocessed512_v2` directory that stores
  the ordinary green channel; the shared generator applies the only inversion.
  Earlier prepared data, caches and audits remain retained.

## Frozen scientific gate

The third-domain gate passes only if all are true:

1. MaskTopo exceeds the highest three-seed-mean strong reducer by at least
   2.0 balanced-accuracy percentage points;
2. MaskTopo minus that same reducer is positive for all three optimization
   seeds;
3. the paired source-image-cluster bootstrap 95% interval for that contrast has
   a lower bound above zero;
4. mean direct structural balanced accuracy is at least 70%;
5. MaskTopo exceeds shuffled topology by at least 3.0 percentage points;
6. the source-image-cluster bootstrap 95% interval for MaskTopo minus shuffled
   topology has a lower bound above zero.

Bootstrap repetitions: 20,000. The strongest reducer is selected by its
three-seed mean, then held fixed for the paired contrast.

## Stopping and interpretation

- Data-gate failure stops training.
- A failed scientific gate is retained and reported; no silent retry or gate
  rewriting is allowed.
- Test labels and performance are already retrospective at the overall project
  level because the method was developed on CrackForest and DeepCrack, but
  FIVES test model performance has not been observed before this frozen run.
- Passing supports cross-domain retinal-vessel connectivity reasoning under
  dense structural supervision. It does not establish clinical utility,
  segmentation state of the art, patient-level generalization, or a universal
  vision connector.
