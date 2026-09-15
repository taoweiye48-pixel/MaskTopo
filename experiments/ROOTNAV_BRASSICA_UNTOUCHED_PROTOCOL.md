# TB-B-260803-034 — RootNav 2.0 Brassica napus untouched external validation protocol

Status: **FROZEN BEFORE ANY BRASSICA IMAGE OR RSML CONTENT ACCESS**  
Frozen date: 2026-08-03 (Asia/Shanghai)  
Experiment ID: `TB-B-260803-034`

## Scientific question and claim boundary

This experiment tests whether `MaskTopo` improves a derived endpoint-connectivity task over equally supervised mask-aware non-topological reducers and a persistent-homology token baseline in a new plant-root structural domain.

- Domain: individual Brassica napus (oilseed rape) root systems on blue germination paper from the RootNav 2.0 author archive.
- This domain is distinct from retinal vessels, pavement cracks and road graphs. No Brassica image or RSML geometry has been opened locally before this freeze.
- Species-only JSON metadata and ZIP central-directory metadata were inspected before freeze; no image pixels, RSML coordinates, labels or model performance were observed.
- This endpoint-connectivity result must never be mixed with the P1–P5 disease-classification `+1.00 pp` result.
- The experiment is one external-domain validation at `K=16`, not a claim of superiority for every domain or K.

## Dataset provenance, license caution and source audit

- Primary publication: Yasrab et al., *RootNav 2.0: Deep learning for automatic navigation of complex plant root architectures*, GigaScience 8(11), 2019, DOI `10.1093/gigascience/giz123`.
- Dataset DOI: `10.5524/100651`.
- Author-institution archive: `https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip`.
- Provenance page: `https://plantimages.nottingham.ac.uk/datasets.html`.
- Terms page: `https://plantimages.nottingham.ac.uk/terms-and-conditions.html`.
- HTTP metadata frozen before content access: Content-Length `2,227,029,019`, ETag `"60c75959-84bdc41b"`, Last-Modified `Mon, 14 Jun 2021 13:27:53 GMT`; server supports byte ranges.
- The paper reports 120 Brassica images with 91 training, 14 validation and 15 holdout test images. The public author archive has no machine-verifiable file-level split labels, so the project split below must not be called the authors' original file-level split.
- Archive numeric IDs 3797 through 3916 contain exactly 120 images, 120 RSML files and 120 species metadata JSON files. Boundary/middle species JSON files identify Oilseed Rape / Brassica Napus.
- Central-directory CRC+uncompressed-size audit found exactly one duplicate image pair: `3797/image_3797.jpg` and `3901/image_3901.jpg`. Before opening either image or RSML, the frozen rule retains the smaller numeric ID 3797 and excludes 3901 without inspecting or comparing RSML content. This yields 119 unique source images.
- Use is restricted to non-commercial research with paper and dataset citation. Dataset-specific terms apply; no stronger license statement will be made than the author-institution terms support.

## Project-defined source split frozen before content access

1. Eligible canonical source IDs are the zero-padded numeric strings `3797` through `3916`, excluding `3901` under the duplicate rule above.
2. For each ID compute `SHA256("TB-B-260803-034|split_seed=20260803|" + source_id)`.
3. Sort ascending by `(digest, source_id)`.
4. Assign the first 90 sources to train, the next 14 to dev and the last 15 to sealed test.
5. Each source maps only to `{id}/image_{id}.jpg` and `{id}/image_{id}.rsml` in the author archive.
6. Freeze the source IDs, archive central-directory CRC/size metadata and code/protocol hashes in a manifest before fetching any image or RSML member.

Split seed: `20260803`. The resulting split is project-defined and source-disjoint. Species JSON metadata may be present locally; image and RSML content must be fetched only according to the frozen split stage.

## Fixed task construction

- Convert RGB to grayscale with `round(0.299 R + 0.587 G + 0.114 B)` to retain the same three-channel query interface used in TB032: grayscale image plus two marker channels.
- Rasterize every RSML root polyline into a binary root mask at the native image resolution with width 8 pixels, matching the RootNav 2.0 paper's ground-truth mask-rendering method. First- and second-order roots are unioned.
- Native crop and model input/output: 64 x 64 pixels; no resize.
- Query endpoints are foreground pixels from eligible 8-connected components in the cropped ground-truth mask. “Endpoint” denotes an endpoint of the query pair, not necessarily a degree-1 skeleton point.
- Positive: both query endpoints belong to the same crop-local 8-connected foreground component.
- Negative: the endpoints belong to different crop-local 8-connected foreground components.
- The two marker channels are rendered identically for both labels. Ground-truth masks create labels and mask-predictor targets only; they are never reducer inputs.
- Minimum eligible component size: 12 pixels.
- Exact balanced counts: train 2400, dev 600, sealed test 1200.
- Candidate multiplier: 3. Data/sample seed: `20260810`, fixed across optimization seeds and methods.
- Apply the existing DeepCrack `matchedv2` shortcut matcher: greedily balance labels jointly within image-intensity quantile cells and endpoint-distance quantile cells. Abort unless exact balance and sufficient matched pairs are achieved.
- Every sample retains its source ID. No source may cross splits.

## Shared mask predictor and float16 path

- Train a shared dense mask predictor using train samples only for 30 epochs; select its checkpoint on dev mask F1 only.
- For each optimization seed, generate one common mask-probability cache for every reducer.
- Save all probabilities as IEEE float16, then reload them for `MaskTopo`, both mask-aware non-topological baselines and both PH candidates. No in-memory float32-only MaskTopo path is permitted.
- Each seed must achieve dev mask F1 at least 0.30 or test stays sealed.

## Equal-supervision model contract

Common settings:

- Exact `K=16` output tokens; token dimension 64.
- Same PatchStem image features, reloaded float16 mask probabilities, endpoint-coordinate markers, samples, augmentation and classifier head.
- Classifier training: 22 epochs, batch size 32, AdamW, learning rate `1e-3`, weight decay `1e-4`; dev balanced accuracy selects checkpoints.
- Optimization seeds: `20260810`, `20260811`, `20260812`.

Reportable methods:

1. `MaskTopo`: connected-component/adjacency-aware reducer using only the common reloaded mask cache.
2. `Mask-Perceiver strong`: TB032 mask-conditioned Perceiver/latent resampler; mask weighting/bias allowed; connected components, component assignment, adjacency/reachability graphs, PH and topology message passing forbidden.
3. `Mask-Slot matched`: TB032 parameter-matched mask-conditioned slot reducer under the same prohibitions and +/-1% parameter-count gate relative to MaskTopo.
4. `best-PH-dev`: train the existing protocol-matched `PH-only` and `PH-guided` candidates from H0/H1 persistent diagrams on the identical float16 mask cache. Select one PH identity using three-seed mean dev balanced accuracy; ties within `1e-12` resolve to `PH-guided`. Freeze the selected identity and all checkpoints before test. Only that identity is evaluated on test.

The PH implementation is a project protocol-matched baseline, not an original-protocol reproduction of a published PH method.

## Train/dev gates before test

All items below must pass and be hash-locked in `实验记录/TB-B-260803-034_train_dev_freeze_manifest.json`:

1. Remote archive identity/size/ETag and central-directory index unchanged.
2. Exactly 119 retained unique image sources, one excluded duplicate ID, and disjoint 90/14/15 project split.
3. Only train/dev image and RSML members fetched before freeze; zero test image/RSML member requests.
4. Exact balanced task counts and matchedv2 shortcut matching.
5. Endpoint-distance-only and marker-only dev balanced accuracies are each at most 0.55.
6. Ground-truth-mask direct structural dev balanced accuracy is at least 0.65.
7. Each seed's mask predictor dev F1 is at least 0.30.
8. Every reducer passes finite-output, identical-input and `[batch,16,64]` exact-token self-tests; non-topological prohibition and parameter gates pass.
9. Three seeds of all reportable model families complete; PH identity is selected using dev only.
10. Protocol, source/sample manifests, code, float16 cache manifests, checkpoints and their SHA-256 values are frozen.

Gate failure is reportable and does not authorize threshold relaxation, source replacement, seed replacement, altered sampling or test access.

## One-shot sealed test

- A separate test-only entry point verifies all frozen hashes before any test member request.
- Immediately before the first test image or RSML member is fetched, atomically create `实验记录/TB-B-260803-034_TEST_INFERENCE_STARTED.json`. If it already exists, refuse to run.
- Fetch exactly the 15 frozen test image/RSML pairs by HTTP range, verify central-directory CRC and size, then generate the frozen 1200 samples.
- Evaluate the four frozen reportable methods and three seeds once. Preserve per-sample labels, scores, predictions and source IDs.
- No test-driven threshold, epoch, PH identity, model, seed or sample change is permitted.

## Statistics and decision language

- Primary metric: balanced accuracy; report all seeds and mean +/- sample standard deviation.
- Primary paired contrasts: `MaskTopo - Mask-Perceiver strong`, `MaskTopo - Mask-Slot matched`, `MaskTopo - best-PH-dev`.
- Paired source-level cluster bootstrap: 20,000 replicates, resampling the 15 test source images with replacement and retaining all samples belonging to each selected source; bootstrap seed `20260803`.
- Strong external fairness support requires both non-topological contrast lower 95% CI bounds above zero.
- Strong external topology-representation support additionally requires the PH contrast lower bound above zero.
- With only 15 test sources, interval width and source heterogeneity must be emphasized. If any CI crosses zero or reverses, report it as inconclusive or negative.
- Do not add more DeepCrack test baselines and do not combine these endpoint-connectivity effects with P1–P5 disease classification.

## Artifact location

All commands, stdout/stderr, manifests, hashes, member requests, model artifacts, one-shot marker, predictions, bootstrap results, failures and reports must be recorded under `E:\TopoBridge_MVP\实验记录`. Large fetched train/dev files may reside under `E:\TopoBridge_MVP\data`, with complete hashes/manifests copied into the experiment record.
