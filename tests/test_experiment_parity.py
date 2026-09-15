"""Check the public API against the original experiment code and checkpoint schema."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from topocoarsen_oracle import TopoCoarsenModel
from crackforest_mechanism_ablation import build_artifacts
from masktopo import MaskTopo, build_topology

class ParityTests(unittest.TestCase):
    def test_artifacts_logits_and_gradients(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(19)
        for dtype in (np.float16,np.float32):
            probs=rng.random((3,64,64)).astype(dtype)
            arrays={'images':np.zeros((3,3,64,64),dtype=np.float32),'endpoint_patches':np.array([[0,255],[10,80],[40,160]]),'connected':np.array([0,1,0])}
            for k in (8,16):
                for closing in (0,1,2):
                    expected,_=build_artifacts(arrays,probs,0.65,closing,k,'mask_topo_recheck',42)
                    actual=build_topology(probs,k,0.65,closing)
                    for key in ('assignment','adjacency','reachability'):
                        np.testing.assert_array_equal(getattr(actual,key),expected[key])
                reference=TopoCoarsenModel('mask_topo_external',64,k).eval()
                public=MaskTopo(k).eval()
                public.load_state_dict(reference.state_dict(),strict=True)
                # Nonzero gates ensure both graph branches are exercised.
                for model in (reference,public):
                    model.head.graph_layers[0].local_gate.data.fill_(0.25)
                    model.head.graph_layers[0].component_gate.data.fill_(-0.15)
                x=torch.randn(3,3,64,64)
                a=x.clone().requires_grad_(True);b=x.clone().requires_grad_(True)
                y=reference(a,*actual.tensors());z=public(b,*actual.tensors())
                torch.testing.assert_close(y,z,rtol=0,atol=0)
                y.sum().backward();z.sum().backward()
                torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

if __name__ == '__main__':unittest.main()
