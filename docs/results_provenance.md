# Reported results

The current manuscript, **MaskTopo: Component-Aligned Token Coarsening for Thin-Structure Connectivity**, is the authoritative source for all reported numerical results in this repository.

[The result files](../reported_results/README.md) transcribe manuscript Tables 1-4. Values, standard deviations and confidence intervals are copied at the precision reported in the paper. Table 3 gains are copied directly from that table rather than recalculated from rounded Table 2 entries.

Protocol labels follow the manuscript: Table 1 contains the primary test comparisons; Tables 2 and 3 use post hoc development data for Massachusetts and Brassica and the historical float32 setup for DeepCrack; Table 4 contains retrospective test ablations. These settings should be interpreted separately.

The result files contain aggregate statistics, not individual seed scores or sample predictions. New evaluations must compute their metrics from the corresponding predictions. Superseded experiment exports are excluded from the current result files.

The exact table excerpts and manuscript checksums accompany the CSVs. Code checks cover model behavior and interfaces; they do not constitute a new full-dataset training run.
