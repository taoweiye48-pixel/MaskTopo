# Reported per-seed results

These CSVs are exported from archived local experiment artifacts. Some manuscript summaries were subsequently updated by the authors; see [result provenance](../docs/results_provenance.md). They are not fresh training runs performed during code release. Balanced accuracy is stored as a fraction in [0,1].

| File | Source / interpretation |
|---|---|
| `fives.csv` | `results_fives_external_summary/source_data.csv`; historical MaskTopo mean 87.36%, preceding the author-updated 89.36% |
| `deepcrack_float16.csv` | `results_deepcrack_float16_alignment/source_data.csv`; use `aligned_masktopo` for Table 1 |
| `deepcrack_mask_aware.csv` | `results_mask_aware_equal_supervision_deepcrack_k16_summary/seed_scores.csv`; matched-mask comparison |
| `deepcrack_phgnet.csv` | tested adaptation of the PHG-Net author encoder |
| `massachusetts.csv` | `results_massroads_external_summary/source_data.csv`; historical MaskTopo mean 74.94%, preceding the author-updated 76.94% |
| `brassica.csv` | per-seed values from `TB-B-260803-035_test_once/RESULT.json`, `aggregates` |

For a model's three-seed summary, multiply the mean and sample standard deviation (`ddof=1`) by 100. Compute gains from unrounded means before rounding when the underlying scores are available. The manuscript's corrected Massachusetts difference (9.80 pp) uses the displayed author-supplied means, as documented in its correction record. These exports do not include the per-source predictions needed for source-cluster confidence intervals.
