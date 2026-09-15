# Experiment guide

## Working directory and environment

Install from the repository root with `python -m pip install -e ".[experiments]"`. Then run the original scripts **from `experiments/`**. Their imports and relative paths retain the original workspace convention.

```bash
cd experiments
python fives_external_gate.py --help
```

The public `masktopo` API can also be used in new projects independently of the historical experiment drivers.

## Training configuration

- Image size: 64×64; PatchStem output: 16×16×64.
- Mask predictor: 30 epochs; downstream classifier: 22 epochs.
- AdamW: learning rate 0.001, weight decay 0.0001, batch size 32.
- Mask threshold, closing and checkpoints are selected on development data.
- Fixed token budgets: FIVES/Massachusetts K=8; DeepCrack/Brassica K=16.
- Standard optimization seeds: 20260810, 20260811, 20260812. Massachusetts uses 20260820, 20260821, 20260822.
- Keep data seeds fixed across optimization seeds. Changing sample construction changes the experiment.

## FIVES

After the preparation described in [datasets.md](datasets.md), run each seed:

```bash
python fives_external_gate.py --dataset-root real_data/FIVES/dataset/preprocessed512_v2 --cache-dir fives_cache_v2 --output results_fives_seed20260810 --train-size 2400 --dev-size 600 --test-size 1200 --candidate-multiplier 3 --crop-size 64 --output-size 64 --min-component-pixels 12 --mask-epochs 30 --epochs 22 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --dim 64 --token-count 8 --num-workers 0 --seed 20260810 --data-seed 20260810 --split-seed 20260731 --shuffle-seed 20260731
```

Repeat with optimization seeds 20260811 and 20260812 and matching output-directory names. The archived runs average 87.36%; the manuscript's later author-supplied 89.36% aggregate has no corresponding updated per-seed predictions in this release. Use `summarize_fives_external.py --help` for aggregation arguments.

## DeepCrack: historical run versus primary comparison

The historical external gate can be run after dataset extraction:

```bash
python deepcrack_external_gate.py --dataset-root real_data/DeepCrack/dataset/extracted --cache-dir deepcrack_cache --output results_deepcrack_seed20260810 --token-count 16 --seed 20260810 --data-seed 20260810 --split-seed 20260730
```

Use all three standard optimization seeds. Additional reducers are run through `strong_reducer_baselines.py` and summarized with the matching summary script.

**This command alone does not produce the primary Table 1 comparison.** The manuscript's main DeepCrack result uses the float16-aligned masks and matched-mask controls:

| Step / analysis | Driver |
|---|---|
| PH-only / PH-guided representations | `ph_token_baselines.py` |
| Float16-aligned MaskTopo evaluation | `deepcrack_float16_alignment.py` |
| Mask-Perceiver / parameter-matched Mask-Slot | `mask_aware_equal_supervision.py` |
| Adapted PHG-Net PointNet encoder | `phgnet_author_adaptation.py` |
| Per-protocol summaries and cluster bootstrap | matching `summarize_*.py` scripts |

These drivers consume stored masks, caches and checkpoints from preceding stages. Inspect their `--help` and corresponding `*_PROTOCOL.md` files for input paths. Some evaluators resolve the original run-directory names: preserve those names or map a local workspace with the same structure. A new optimizer seed is not a new data seed.

The native MaskTopo probabilities in `deepcrack_float16.csv` give 74.94% before storage alignment; the **aligned** rows give the manuscript's **74.97%**. The manuscript's historical float32 correspondence/branch ablations use an author-updated **78.94%** summary; the archived runs contain **74.94%**. Neither belongs in Table 1. See [result provenance](results_provenance.md) for the unavailable updated predictions.

## Massachusetts Roads

```bash
python massroads_data_gate.py --dataset-root real_data/MassachusettsRoads --cache-dir massroads_cache --output results_massroads_data_audit
python massroads_train_dev.py --dataset-root real_data/MassachusettsRoads --cache-dir massroads_cache --data-audit results_massroads_data_audit --output results_massroads_seed20260820_train_dev_retry1 --seed 20260820
```

Run seeds 20260821 and 20260822 with outputs `results_massroads_seed20260821_train_dev` and `results_massroads_seed20260822_train_dev`. The `retry1` name for the first seed is retained because the historical freeze script expects it. It is a directory label, not an instruction to repeat a failed result selectively.

Then inspect/run `massroads_freeze_manifest.py` and the `--help` stages of `massroads_test_once.py`. The included freeze manifest is a provenance record of the original checkpoints. A newly trained run requires its own verified artifact record. The archived MaskTopo test mean is 74.94%; the manuscript's later author-supplied summary is 76.94% (see [provenance](results_provenance.md)). The primary test comparison is matched G2TM; the post hoc development correspondence study is `massroads_dev_mechanism.py`.

## RootNav2 Brassica

The scripts retain the original source-archive split, deduplication and endpoint-marker matching workflow:

1. `brassica_untouched_prepare.py --stage manifest`
2. `brassica_untouched_prepare.py --stage extract-train-dev`
3. `brassica_covariate_gate.py` to construct the matched-v3 samples.
4. `brassica_train_dev.py --seed SEED --output OUTPUT` for each standard seed.
5. `brassica_freeze_train_dev.py` to verify the training/development artifacts.
6. `brassica_test_once.py --stage preflight`, followed by `--stage run-once` only for the corresponding verified workspace.

Read `ROOTNAV_BRASSICA_UNTOUCHED_PROTOCOL.md` and its V2 revision in `experiments/`. The historical test driver checks archive ETags, checksums, manifests and one-shot markers. These checks are part of the study protocol. The included historical manifests reference original paths and checkpoints; the public release does not include those checkpoints, and an independent retraining will generally have different weight hashes. Consequently, the archived one-shot scripts are **not a turnkey claim of bit-identical retraining on arbitrary machines**. The reusable model, preparation, training and evaluation logic are provided; use new provenance records for new runs and report them as such.

The development correspondence ablation is `brassica_dev_mechanism.py`; its development scores are distinct from the 89.14% primary test score.

## Ablations and additional analyses

| Analysis | Entry points |
|---|---|
| Assignment / grid / identity / shuffled controls | `crackforest_mechanism_ablation.py`, FIVES driver, Massachusetts/Brassica development drivers |
| Adjacency / reachability branches | `ar_branch_ablation.py`, `summarize_ar_branch_ablation.py` |
| K sweep | `masktopo_token_budget.py`, `ph_token_baselines.py`, `summarize_token_budget.py` |
| Mask perturbations | `robustness_stress.py`, `summarize_robustness_stress.py` |
| Disease-classification pilot | `p0_full_finetune.py` |
| Road-network pilot | `spacenet3_stage1_run.py` and associated preparation/metric files |

`Assignment+identity` retains branch transformations with identity A/R; `Assignment only` disables both branches. These controls are not interchangeable. A/R ablation standard deviations describe seed variability; the manuscript did not have source-level predictions for corresponding cluster-bootstrap intervals.

## Release verification and limits

The release was checked with core tests and parity tests against the unmodified experiment code, using K=8/16, float16/float32 probabilities, closing values 0/1/2, shared classifier weights, and nonzero graph gates. Full dataset training was not rerun during repository preparation. The API demo uses random weights and synthetic data. Dataset images, full generated sample arrays and trained weights are not distributed with this source release.

Preserve the dataset source as the resampling unit when reproducing source-cluster confidence intervals. Per-seed result CSVs alone do not contain the per-source predictions needed to recompute those intervals.
