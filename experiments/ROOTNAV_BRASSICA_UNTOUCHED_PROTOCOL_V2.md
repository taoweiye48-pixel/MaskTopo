# TB-B-260803-035 — Brassica untouched validation after train/dev shortcut repair

Status: **FROZEN BEFORE ANY TEST IMAGE OR TEST RSML ACCESS**  
Frozen date: 2026-08-03 (Asia/Shanghai)  
Experiment ID: `TB-B-260803-035`

## Why a new experiment ID is required

`TB-B-260803-034` was a train/dev-only pilot under the frozen DeepCrack `matchedv2` sampler. It passed source integrity, balance, endpoint-distance, intensity, ground-truth structural and array-invariant gates, but failed its pre-test marker-only shortcut gate: deterministic marker-coordinate logistic dev balanced accuracy was 64.00%, above the frozen 55% ceiling. No Brassica test image or RSML member was requested.

TB034 is terminated and its matcher may not be retroactively changed. TB035 is a newly pre-specified confirmatory experiment that retains the same already-frozen source partition and sealed test IDs but replaces task-sample matching before test access. This dev-informed protocol evolution must be disclosed; it is not a claim that the complete task protocol was chosen without train/dev inspection.

## Unchanged base protocol

All dataset provenance, license caution, source-level duplicate rule, source split, image/RSML rendering, task definition, mask predictor, float16 path, model identities, optimizer budget, seeds, train/dev gates, one-shot test firewall, statistics, artifact location and claim boundaries from `ROOTNAV_BRASSICA_UNTOUCHED_PROTOCOL.md` remain binding except where explicitly replaced below.

- Base protocol SHA-256: `d9725bff3f3865dd7e1f3f6c5bcb4aba6e10eafaccdfe53d58eeb0a1936f979f`.
- Frozen source-manifest SHA-256: `102ad36ce884045141cf5b71919667fa81a09e2d56829ebbc60022be0ae84de0`.
- Project-defined split remains 90 train / 14 dev / 15 sealed test.
- Sealed test IDs remain exactly: `3911, 3868, 3899, 3860, 3848, 3797, 3798, 3887, 3808, 3852, 3905, 3863, 3829, 3886, 3889`.
- Test image requests = 0; test RSML requests = 0 at this freeze.
- Data/sample seed remains `20260810`; train/dev/test reportable sample counts remain 2400/600/1200.
- Model training remains K=16, 22 epochs, batch 32, AdamW lr `1e-3`, weight decay `1e-4`, seeds 20260810/11/12; mask predictor remains 30 epochs.

## Replacement task matcher: marker-covariate matchedv3

Generate the same deterministic candidate pools as TB034 with candidate multiplier 3: 7200 train candidates, 1800 dev candidates and, only after the one-shot test marker is created, 3600 test candidates. Each pool is exactly label-balanced before matching.

For every candidate, compute the following frozen covariates without using model output:

1. normalized row and column centroids for marker 1;
2. normalized row and column centroids for marker 2;
3. signed row and column deltas;
4. absolute row and column deltas;
5. Euclidean marker distance;
6. midpoint row and column;
7. within-marker row×column products for markers 1 and 2;
8. signed-delta product;
9. precise generator endpoint distance;
10. mean grayscale intensity.

This yields 16 covariates. Within each split independently:

1. Standardize all 16 covariates using that candidate pool's unlabelled mean and standard deviation. Zero-variance dimensions use scale 1.
2. Treat negative candidates as query points and positive candidates as the reference bank.
3. Use exact Euclidean k-nearest neighbours with `k=min(256, number_of_positive_candidates)` and deterministic brute-force search.
4. Create edges `(distance, negative_index, positive_index)` for all returned neighbours; sort ascending lexicographically by those three values.
5. Greedily accept an edge only if neither candidate has already been used, until exactly half the reportable sample count pairs have been selected.
6. Abort if insufficient unique pairs exist. Flatten each accepted pair as negative then positive, then apply one deterministic shuffle with seed `data_seed + 77,777` for train, `data_seed + 1,077,777` for dev and `data_seed + 2,077,777` for test.

The matcher uses labels only to create the already-required balanced paired sample set; it does not use any reducer prediction, dev model metric or test aggregate. No propensity, threshold or k selection is allowed after viewing matchedv3 dev results.

## Re-frozen pre-test task gates

Before any neural-network training and before test access, matchedv3 train/dev must satisfy all TB034 gates, including:

- exact 2400/600 balance and source membership;
- endpoint-distance-only dev balanced accuracy at most 0.55;
- the unchanged StandardScaler + class-balanced logistic regression (`C=1`, random state 20260803) on the 14 marker-coordinate features must achieve dev balanced accuracy at most 0.55;
- mean-intensity shortcut is reported but is not a separate hard gate;
- ground-truth direct structural dev balanced accuracy at least 0.65;
- all array invariants pass.

If matchedv3 fails, TB035 terminates and test stays sealed. The matcher, `k`, feature set, counts and gate thresholds may not be modified within TB035.

## Confirmatory interpretation

If all train/dev gates pass and the subsequent one-shot test supports the frozen contrasts, the result may be described as an untouched-test external validation after a disclosed train/dev task-shortcut repair. It must not be described as an entirely untouched end-to-end protocol selection. Endpoint-connectivity effects remain strictly separate from P1–P5 disease classification `+1.00 pp`.

