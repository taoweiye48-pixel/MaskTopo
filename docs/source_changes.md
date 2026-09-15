# Release source changes

The experiment implementations and frozen protocol documents originate from the research workspace. `original_source_sha256.json` records their original hashes; `release_source_sha256.json` records the corresponding files in this release.

- `experiments/massroads_dev_mechanism.py`: removed an obsolete numerical test-gain reference from its module description and generated report explanation. The text now refers to the primary test result in the manuscript. Model, training, evaluation and metric calculations are unchanged.

Reported numerical results are taken from the current manuscript, as described in [results_provenance.md](results_provenance.md).
