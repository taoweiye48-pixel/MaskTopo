"""Synthetic interface demo with random weights; not an accuracy experiment."""
import numpy as np
import torch
from masktopo import MaskTopo, build_topology

torch.manual_seed(0)
torch.set_num_threads(1)
images = torch.zeros(2, 3, 64, 64)
images[:, 0, 8:56, 16] = 1
images[:, 0, 8:56, 48] = 1
images[:, 1, 12, 16] = 1
images[0, 2, 48, 16] = 1
images[1, 2, 48, 48] = 1
# Synthetic probability maps isolate the topology API. Real inference uses the
# output of a trained image-only MaskPredictor, never the annotation mask.
probabilities = images[:, 0].numpy() * 0.9
for k in (8, 16):
    topology = build_topology(probabilities, token_count=k, threshold=0.5)
    model = MaskTopo(token_count=k).eval()
    with torch.no_grad():
        logits = model(images, *topology.tensors())
    assert logits.shape == (2,) and torch.isfinite(logits).all()
    assert all(np.unique(a).size == k for a in topology.assignment)
    print(f"K={k}: exact nonempty tokens; logits shape={tuple(logits.shape)}; parameters={sum(p.numel() for p in model.parameters()):,}")
print("Smoke demo passed. Random weights do not represent trained accuracy.")
