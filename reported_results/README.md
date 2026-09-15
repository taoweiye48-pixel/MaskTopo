# Results from the manuscript

**All numerical results here follow the current manuscript.** These files contain reported aggregates from Tables 1-4. BA means and standard deviations use percent (%); differences and confidence intervals use percentage points (pp).

| File | Manuscript source | Contents |
|---|---|---|
| [table1_primary_results.csv](table1_primary_results.csv) | Table 1 | Four primary test results, MaskTopo mean/SD, comparator means, gains and source-cluster 95% CIs |
| [table2_component_alignment.csv](table2_component_alignment.csv) | Table 2 | Component-alignment and graph-token correspondence means |
| [table3_graph_correspondence_gains.csv](table3_graph_correspondence_gains.csv) | Table 3 | Graph and correspondence gains, copied as reported |
| [table4_adjacency_reachability.csv](table4_adjacency_reachability.csv) | Table 4 | Adjacency/reachability ablation means, SDs and parameter counts |
| [manuscript_tables.tex](manuscript_tables.tex) | Exact table excerpts | Captions, entries and table notes, including additional comparator summaries |
| [manuscript_source.json](manuscript_source.json) | Source metadata | Manuscript title and LaTeX/PDF checksums |

The Massachusetts and Brassica analyses in Tables 2-3 use post hoc development data. DeepCrack's float32 ablations are distinct from the primary float16 comparison in Table 1. All Table 4 evaluations are retrospective, as stated in the paper.

Individual seed observations are not supplied or inferred from these summaries. Table 4 reports seed variability only; the paper states that the source-level predictions needed for cluster-bootstrap intervals are unavailable.

## Refresh the exports

From the repository root, regenerate the CSVs from the included table excerpts:

```bash
python tools/export_manuscript_results.py
```

When revising the manuscript, refresh the excerpts and record the matching source checksums:

```bash
python tools/export_manuscript_results.py --manuscript /path/to/main.tex --pdf /path/to/main.pdf
```

See [reported-result conventions](../docs/results_provenance.md).
