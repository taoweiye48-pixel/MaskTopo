# TopoBridge Massachusetts Roads untouched external confirmation protocol

Frozen on: 2026-08-02, before downloading or opening any official test image or target map and before any Massachusetts Roads model performance is produced.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Verification Status: PROTOCOL_FROZEN_BEFORE_TEST_ACCESS
- Version Label: massroads_protocol_v1.2
- Audit record: `实验记录/TB-B-260802-020_路线二证据矩阵与缺口审计.md`

### v1.2 implementation clarification (2026-08-02)

The first seed stopped immediately after mask-predictor training because a legacy CrackForest grid helper hard-coded a 4×4 reshape and therefore could not instantiate the already frozen K=8 budget. No endpoint-connectivity model was trained, no endpoint-connectivity dev comparison was produced, and official test files remained absent. Before any retry, `grid_external` is fixed to the existing FIVES rectangular-grid rule: choose the closest admissible factorization with both factors dividing 16. At K=8 this is a 2×4 grid, expanded over the 16×16 fine-token lattice so that every reduced token contains exactly 32 fine tokens. This clarification changes no sample, mask threshold, seed, optimizer, epoch, method list, statistical gate, or test-access rule.

## Research question

Does topology-supervised token coarsening retain an advantage over strong fixed-budget reducers on a previously untouched aerial-road domain when the official test split is evaluated exactly once?

The target is a derived road endpoint-connectivity task. Passing this experiment supports cross-domain thin-structure connectivity reasoning; it does not by itself remove the broader limitation that endpoint labels are constructed from dense masks.

## Dataset, split, and license boundary

- Dataset: Massachusetts Roads Dataset, distributed by Volodymyr Mnih at `https://www.cs.toronto.edu/~vmnih/data/`.
- Official file-index counts: 1108 train, 14 validation, and 49 test image/target pairs.
- Input and target resolution: 1500×1500 pixels; imagery resolution is reported as 1 m/pixel in the dataset description/thesis.
- The official train/validation/test split is used without reassignment.
- The distribution page instructs research users to cite Mnih's 2013 thesis but exposes no standardized CC/SPDX license. Status is therefore `RESEARCH_USE_CAUTION`; data and derivatives remain local and are not redistributed.
- Train and validation may be downloaded and audited immediately after this protocol is frozen.
- Official test images and target maps must not be downloaded, opened, rendered, enumerated beyond public filename metadata, or used by code until all train/validation data gates, task rules, model list, seeds, and decision gates below are fixed and pass.

## Frozen derived task

For each 128×128 native-resolution aerial crop, place two visible endpoint markers on annotated road pixels and predict whether they belong to the same 8-connected component in the official raster target map.

- The RGB crop is converted to fixed ITU-R BT.601 luminance, resized to 64×64 with Lanczos interpolation, and inverted exactly once to match the existing thin-structure pipeline.
- The 128×128 binary road target used to train the mask predictor is reduced to 64×64 by fixed non-overlapping 2×2 max pooling, preserving annotated road pixels. Connectivity labels remain defined on the native 128×128 target before this reduction.
- Connectivity labels and component membership are computed on the native 128×128 target crop before image resizing.
- A positive pair is drawn from one eligible connected component; a negative pair is drawn from two distinct eligible components.
- Endpoints must occupy distinct cells of the fixed 16×16 fine-token grid.
- Eligible components must contain at least 32 native target pixels.
- Requested endpoint distances cycle deterministically through normalized values 0.20, 0.34, and 0.48, using the existing maximum tolerance 0.075.
- Candidate counts are three times the requested final sample count.
- Final samples are exactly class-balanced and matched using train-derived fixed quantile edges for mean luminance, endpoint distance, and target-road fraction. The fixed grid is 4×3×3 bins. No split-specific bin relaxation is allowed.
- Final counts: 4000 train, 600 validation, 1200 test.
- Statistical unit: official source tile, not crop and not optimization seed.

Known construct limitations to retain: rasterized crossings can merge overpasses; crop boundaries can split a road network; matched 50/50 prevalence is not deployment prevalence.

## Frozen models and fairness contract

All methods use the same 64×64 input, PatchStem, downstream classifier, optimizer, epoch budget, dev-only checkpoint selection, and exact K=8 reduced-token budget.

Primary method:

- `mask_topo_external`: predicted road mask → component-aligned pooling → aligned graph messages.

Strong reducer controls:

- `grid_external`, using the frozen rectangular-grid rule above (2×4 at K=8; 32 fine tokens per reduced token);
- `tokenlearner`;
- `perceiver_resampler`;
- `tome_style`;
- `g2tm_fixedk_mask`, explicitly labeled a matched fixed-budget adaptation rather than an official Segmenter reproduction.

Mechanism controls:

- `mask_assignment_identity`;
- `shuffled_topology`.

The mask predictor is trained only on official train target maps. Its threshold, closing iterations, and checkpoint are selected only on official validation. The same saved predicted-mask probabilities are used by MaskTopo and mask-guided G2TM. No method receives a ground-truth test mask or topology at inference.

## Frozen optimization

- Data seed: `20260802`.
- Optimization seeds: `20260820`, `20260821`, `20260822`.
- Mask predictor: 30 epochs.
- Downstream models: 22 epochs.
- Batch size: 32.
- AdamW learning rate: 1e-3.
- Weight decay: 1e-4.
- Classification threshold: fixed 0.5.
- Mask thresholds: 0.1 through 0.9 in steps of 0.1, selected on validation only.
- Morphological closing iterations: 0, 1, 2, selected on validation only.

## Train/validation data gate before test access

Test download and model evaluation are prohibited unless all conditions pass on train/validation:

1. Official file counts and image/target name pairing are exact for downloaded train and validation files.
2. Train and validation source identifiers are disjoint.
3. Every file is 1500×1500 and every target is binary after the fixed `>=128` conversion.
4. Final train/validation samples are exactly 50/50 positive/negative.
5. At least 800/1108 train tiles and all 14 validation tiles contribute samples.
6. A train-fitted one-dimensional threshold shortcut is at most 55% balanced accuracy on validation for each of endpoint distance, mean luminance, and target-road fraction.
7. Endpoint markers overlap positive target pixels, endpoint patch IDs agree with the label, and all saved shapes/dtypes are invariant.
8. At least 600 matched test pairs are projected to be feasible from validation candidate yield; this is a feasibility calculation only and does not access test.

If a gate fails, no test file is downloaded. A generic correction may be proposed only from train/validation, with the failed v1 artifacts retained and a new protocol version frozen before proceeding.

## Test data integrity gate

After the train/validation gate passes, download the official test split once and verify only:

1. 49 image/target pairs, exact filename pairing, 1500×1500 dimensions, and binary targets;
2. no source identifier overlap with train/validation;
3. deterministic fixed-rule generation of exactly 1200 balanced samples;
4. at least 45/49 test tiles contribute samples;
5. marker/mask/component-label invariants and finite arrays.

Failure stops before any model test inference. Test shortcut accuracies are reported after the one-shot evaluation but cannot trigger task or model revision.

## Frozen scientific gate

The experiment passes only if all are true:

1. MaskTopo exceeds the highest three-seed-mean strong reducer by at least `+2.0 pp`.
2. MaskTopo minus that fixed strongest reducer is positive for all three seeds.
3. The paired source-tile cluster bootstrap 95% CI lower bound for that contrast is above zero.
4. MaskTopo exceeds assignment-only by at least `+1.0 pp`, with source-tile cluster CI lower bound above zero.
5. MaskTopo exceeds shuffled topology by at least `+2.0 pp`, with source-tile cluster CI lower bound above zero.
6. Mean direct structural balanced accuracy is at least 70%.

Bootstrap repetitions: 20,000. The strongest reducer is selected by the frozen three-seed mean among Grid, TokenLearner, Perceiver, ToMe-style, and matched G2TM, then held fixed for the paired contrast.

## Stopping and downgrade rules

- No silent retry, seed replacement, test-conditioned task revision, or gate rewriting.
- If MaskTopo is explained by assignment-only or tied by a strong reducer, downgrade the novelty/causal mechanism claim.
- If the one-shot test fails while dev passes, retain the failure and stop this external-domain branch.
- If only the endpoint task benefits, retain the limitation that natural road-network downstream utility remains unproven.

## Required artifacts

- License/source snapshot metadata and official filename manifests.
- Per-file SHA-256, byte size, dimensions, and split manifest.
- Frozen command lines and environment.
- Train/validation data-gate JSON and log before test access.
- Test integrity JSON, predictions, checkpoints and hashes.
- Source-tile cluster bootstrap contrasts and all failed points.
- Experiment record and index update under `E:\TopoBridge_MVP\实验记录`.

## Pre-performance protocol clarification

- v1.1 was frozen on 2026-08-02 before any Massachusetts model performance and before any official test file access.
- v1 specified image resizing but omitted the exact dense-target reduction operator. v1.1 fixes it to non-overlapping 2×2 max pooling. This is a representation clarification, not a result-conditioned change; v1 remains recoverable from the experiment log history.
