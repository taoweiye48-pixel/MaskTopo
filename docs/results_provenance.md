# Result provenance

The manuscript and the archived experiment exports have distinct revision histories. The release preserves the original per-seed values; it does not reconstruct missing seeds by shifting historical scores.

| Setting | Archived MaskTopo mean ± SD (%) | Current manuscript mean ± SD (%) | Evidence available in this release |
|---|---:|---:|---|
| FIVES primary test | 87.36 ± 2.79 | 89.36 ± 2.79 | Earlier per-seed CSV; later author-supplied aggregate only |
| Massachusetts primary test | 74.94 ± 0.76 | 76.94 ± 0.76 | Earlier per-seed CSV; later author-supplied aggregate only |
| DeepCrack historical float32 / A+R | 74.94 ± 2.64 | 78.94 ± 2.64 | Original analysis code; later author-supplied aggregate only |
| DeepCrack aligned float16 primary test | 74.97 ± 2.60 | 74.97 ± 2.60 | Matching per-seed CSV |
| Brassica primary test | 89.14 ± 0.81 | 89.14 ± 0.81 | Matching per-seed CSV |

The local author-update records are dated 2026-08-10: `TB-B-260810-042` records the aggregate updates; `TB-B-260810-043` corrects the Massachusetts mean to 76.94%. Source-record hashes are recorded in [author_update_provenance.json](author_update_provenance.json).

Updated per-seed scores, checkpoints, and per-source predictions corresponding to the changed aggregates were not available during release preparation. The README reproduces the current manuscript summaries and marks the affected primary rows; the archived CSVs must not be cited as reproducing those updated values. Likewise, the reported updated confidence intervals cannot be independently recomputed from the historical predictions or from per-seed summaries alone.

The Massachusetts manuscript gain of 9.80 pp is the difference of the displayed updated means, 76.94 - 67.14. It is not a recomputation from updated unrounded seed scores. The DeepCrack float16 and Brassica primary comparisons remain supported by their matching archived seed exports.

Code verification covers structural invariants, original-implementation parity, and executable interfaces. It does not independently validate the author-supplied aggregate updates or constitute a rerun of full training.
