# TB-B-260803-033 — RootNav 2.0 Arabidopsis untouched external validation protocol

Status: **FROZEN BEFORE ANY IMAGE/MASK PIXEL ACCESS; v1.1 administrative correction after archive-name-only audit**  
Frozen date: 2026-08-03 (Asia/Shanghai)  
Experiment ID: `TB-B-260803-033`

Protocol amendment trail: v1 SHA-256 `ce3fde04d5fa5e919a97887782a5389ce7eeade6a95ea0c723faead4fe9b3af5`. Before any TIFF or RSML content was opened, v1.1 corrected the crop/learning-rate fields to the actual TB032 frozen values and clarified that “endpoint” means the two query endpoints, not degree-1 skeleton pixels. The correction used only existing code/results and ZIP member names, not RootNav pixels, labels or metrics.

## Scientific question and claim boundary

This experiment tests whether the endpoint-connectivity advantage of `MaskTopo` over equally supervised mask-aware non-topological reducers and a persistent-homology token baseline transfers to a previously untouched structural domain.

- Domain: Arabidopsis root architecture from RootNav 2.0 (`RootNav2A`).
- Task: a derived balanced binary endpoint-connectivity task within source images.
- This is not a disease-classification experiment. Its effect sizes must never be combined with or described as the P1–P5 disease-classification `+1.00 pp` result.
- The experiment cannot establish superiority on every domain or every K. It is one pre-specified external-domain validation at `K=16`.
- FIVES, DeepCrack, CrackForest, Massachusetts Roads and SpaceNet 3 are not eligible for further untouched claims because their test data or aggregate test behavior has already been observed locally.

## Dataset identity, provenance and permitted use

- Dataset: RootNav 2.0 Arabidopsis (`RootNav2A`), 277 near-infrared root images with semantic masks, native resolution 1024 x 1024.
- Primary publication: Yasrab et al., *RootNav 2.0: Deep learning for automatic navigation of complex plant root architectures*, GigaScience 8(11), 2019, DOI `10.1093/gigascience/giz123`.
- Dataset DOI: `10.5524/100651`.
- Download transport selected before access: CNCB-NGDC OPIA mirror, `https://download.cncb.ac.cn/OPIA/RootNav2_Arabidopsis.zip`.
- Provenance page: `https://ngdc.cncb.ac.cn/opia/dataset/datasets?dataId=21`.
- Author-institution terms page: `https://plantimages.nottingham.ac.uk/terms-and-conditions.html`.
- Use is restricted to this non-commercial research validation, with paper and dataset citation. The Nottingham terms state that dataset-specific terms apply and that datasets are typically CC-BY-NC; no stronger license claim will be made unless the downloaded archive provides one.
- The downloaded archive SHA-256, byte size, member listing and any embedded license/readme text will be recorded before extraction.

## Source split frozen before pixel access

The paper reports 200 training, 27 validation and 50 holdout-test images. A file-level author split will be used only if the archive contains an unambiguous, machine-verifiable split manifest or split directories that exactly yield 200/27/50 paired sources.

Otherwise, the following project-defined deterministic split is frozen now and must be labelled as such:

1. Enumerate only ZIP central-directory member names; do not decode image or mask pixels.
2. Pair one image and one semantic mask per canonical source ID. Abort unless there are exactly 277 unique pairs.
3. Normalize each source ID to lower-case POSIX-style relative text without the role-specific image/mask directory or filename suffix.
4. Compute `SHA256("TB-B-260803-033|split_seed=20260803|" + canonical_source_id)`.
5. Sort ascending by `(digest, canonical_source_id)`.
6. Assign the first 200 sources to train, the next 27 to dev and the final 50 to sealed test.
7. Store source IDs, archive members and digests in a manifest. The manifest may expose test filenames and metadata but no decoded test pixels, dimensions, class proportions, quality summaries or examples.

No source may occur in more than one split. Split seed: `20260803`.

## Fixed task construction

- Native crop and model input: 64 x 64 pixels, identical to the TB032 crop/output setting; no spatial resize is applied.
- Foreground: the union of all root semantic classes; background excluded. Any archive-specific color/class decoding must be inferred using train only, recorded, and then frozen before dev evaluation. It may not be changed after viewing dev metrics.
- Query-endpoint candidates: foreground pixels of eligible 8-connected components in the binary ground-truth root mask, matching the existing endpoint-connectivity generator. “Endpoint” here denotes an endpoint of the queried pair, not necessarily a degree-1 endpoint of the root skeleton.
- Positive: the two sampled endpoint markers belong to the same 8-connected foreground component.
- Negative: the two sampled endpoint markers belong to different 8-connected foreground components.
- Endpoint markers are rendered identically for both labels. The image crop, two endpoint-coordinate channels and the predicted foreground-mask probability are the common model inputs.
- Exact balanced sample counts: train 2400, dev 600, sealed test 1200.
- Sample-generation seed: `20260803`, fixed across optimization seeds and all methods.
- Shortcut matching: positive and negative samples are greedily matched within split for endpoint Euclidean-distance decile and foreground-fraction decile. Abort if exact balance or matching cannot be achieved.
- Source ID is retained for every sample. No crop derived from a source outside the corresponding source split is permitted.

## Fixed mask prediction and numeric path

- A shared mask predictor is trained only on train sources for 30 epochs; dev is used only for checkpoint selection.
- For each optimization seed, the selected mask predictor produces a single common probability cache for every method.
- All probability caches are saved as IEEE float16 before tokenization. `MaskTopo`, mask-aware non-topological baselines and PH baselines all reload the same saved float16 values; no in-memory float32 exception is allowed.
- Ground-truth masks are used only to generate task labels and to audit mask quality; they are not reducer inputs.
- Mask-predictor dev F1 gate: at least 0.30 for every seed. If it fails, test stays sealed.

## Frozen models and equal-supervision contract

Common settings:

- `K=16` exact output tokens for every reducer.
- Same image features, float16 mask probabilities, endpoint coordinates, classifier head, training samples and augmentation.
- Classifier training: 22 epochs, batch size 32, AdamW, learning rate `1e-3`, weight decay `1e-4`; dev balanced accuracy selects the checkpoint. These are the TB032 frozen optimizer settings.
- Optimization seeds: `20260830`, `20260831`, `20260832`.
- Data and sample identities remain identical across these seeds.

Tested methods:

1. `MaskTopo`: connected-component/adjacency-aware topology token reducer, using only the reloaded float16 cache.
2. `Mask-Perceiver strong`: mask-conditioned Perceiver/latent resampler from TB032; mask attention or weighting allowed; connected components, adjacency graphs, PH and topology message passing forbidden.
3. `Mask-Slot matched`: parameter-matched mask-conditioned slot resampler from TB032 under the same prohibitions.
4. `best-PH-dev`: train both the existing protocol-matched `PH-only` and `PH-guided` candidates on train/dev. Before test, select exactly one PH identity using the three-seed mean dev balanced accuracy; ties within `1e-12` resolve to `PH-guided`. Freeze that identity and all three checkpoints. Only the selected PH identity may be evaluated on test. It must not be described as a published-method original-protocol reproduction.

All methods must pass a structural self-test confirming output shape `[batch, 16, token_dim]`, finite outputs, identical input sample IDs and the prohibition checks above.

## Train/dev gates before test may be opened

The test remains sealed unless all of the following are satisfied and written into a hash-locked train/dev freeze manifest:

1. Dataset/license/provenance audit and archive SHA-256 are complete.
2. Exactly 277 image-mask source pairs and a disjoint 200/27/50 manifest are frozen.
3. Train/dev task generation reaches the exact balanced counts and shortcut matching above.
4. Endpoint-distance-only and marker-only dev balanced accuracies are each at most 0.55.
5. A ground-truth-mask direct structural probe reaches dev balanced accuracy at least 0.65.
6. All three mask predictors reach dev mask F1 at least 0.30.
7. All reducer self-tests, exact-K checks and equal-input checks pass.
8. Three seeds of all four reportable methods are complete. Both PH candidates are trained, but only the dev-selected identity is reportable on test.
9. Code files, protocol, source manifest, sample manifest, float16 cache manifests, selected checkpoints and their SHA-256 values are written to `TB-B-260803-033_train_dev_freeze_manifest.json`.

Failing a frozen gate is a reportable result. It does not authorize threshold relaxation, seed replacement, altered sampling or test access.

## One-shot sealed-test procedure

- A separate test-only entry point must verify every frozen hash before opening or extracting any test image/mask member.
- Immediately before the first test pixel is decoded, atomically create `实验记录/TB-B-260803-033_TEST_INFERENCE_STARTED.json`. If that marker already exists, the program must refuse to run.
- Generate exactly the frozen 1200 balanced test samples using the frozen code and seed.
- Run each frozen seed checkpoint for the four reportable methods once. No threshold, model, epoch, seed or PH identity may be changed afterward.
- Preserve per-sample labels, predictions, scores and source IDs. No subsequent additional DeepCrack-test baseline is part of this experiment.

## Frozen statistics and decision language

- Primary metric: balanced accuracy.
- Report each seed and the three-seed mean +/- sample standard deviation.
- Primary paired contrasts: `MaskTopo - Mask-Perceiver strong`, `MaskTopo - Mask-Slot matched`, and `MaskTopo - best-PH-dev`.
- Confidence intervals: paired source-level cluster bootstrap, 20,000 replicates, resampling the 50 test source images with replacement and retaining all samples belonging to each selected source; bootstrap seed `20260803`.
- Strong external fairness support requires the lower 95% CI bound to exceed zero for both non-topological mask-aware contrasts.
- Strong external topology-representation support additionally requires the lower 95% CI bound to exceed zero for the dev-selected PH contrast.
- Otherwise report the result as inconclusive or negative as observed. Never substitute a sample-level CI.

## Artifact location

All commands, stdout/stderr, manifests, hashes, checkpoints, one-shot marker, predictions, bootstrap output, failures and the final report must be recorded under:

`E:\TopoBridge_MVP\实验记录`

Large data and model artifacts may reside elsewhere under `E:\TopoBridge_MVP`, but their manifests and SHA-256 hashes must be copied into the experiment record.
