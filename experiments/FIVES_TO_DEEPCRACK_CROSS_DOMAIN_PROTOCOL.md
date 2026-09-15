# FIVES to DeepCrack frozen cross-domain protocol

This is a transfer diagnostic. The source models are trained on FIVES and are
evaluated directly on DeepCrack without target-domain dense-mask fine-tuning and
without target-domain classifier training.

## Frozen source

- Three source seeds: 20260810, 20260811, 20260812.
- Source mask threshold and closing are read from the corresponding FIVES
  development-only calibration.
- Source checkpoints are loaded from the completed FIVES runs.

## Target handling

- DeepCrack split and crop cache are fixed before evaluation.
- A FIVES-trained mask predictor produces target mask probabilities.
- Predicted target probabilities alone construct the MaskTopo assignments and
  quotient graph. DeepCrack ground-truth masks are not read during construction.
- DeepCrack labels are used only to calculate final balanced accuracy and the
  target mask F1 diagnostic.

## Models

- Frozen grid topology classifier.
- Frozen MaskTopo classifier.
- Frozen TokenLearner, ToMe-style, mask-guided query reducers.
- Frozen matched G2TM-FixedK feature and mask reducers.

The principal question is whether MaskTopo retains a positive advantage over
grid and non-topological reducers under an unseen domain without target
dense-mask training. This does not replace a genuinely independent downstream
task, because the endpoint-connectivity label definition remains shared.
