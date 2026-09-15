import unittest
import numpy as np
import torch
from masktopo import MaskTopo, MaskPredictor, GatedTopologicalMessaging, build_topology

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_partition_graph_and_reachability(self):
        probabilities = np.zeros((4,64,64), dtype=np.float32)
        probabilities[1] = 1
        probabilities[2, ::8, ::8] = 1
        probabilities[3, 5:59, 10] = 1
        probabilities[3, 5:59, 44] = 1
        for k in (1,8,16,256):
            top = build_topology(probabilities, token_count=k)
            for groups,assignment,a,r in zip(top.groups,top.assignment,top.adjacency,top.reachability):
                self.assertEqual(np.unique(assignment).size,k)
                self.assertTrue(np.array_equal(a,a.T))
                self.assertTrue(np.diag(a).all())
                self.assertTrue(np.all(a <= r))
                self.assertTrue(np.array_equal(r,r.T))
                self.assertTrue(np.array_equal((r.astype(np.int32) @ r.astype(np.int32)) > 0, r.astype(bool)))
                for token in range(k):
                    self.assertEqual(np.unique(groups[assignment == token]).size,1)

    def test_zero_gates_identity_and_gradient(self):
        layer = GatedTopologicalMessaging(8)
        h = torch.randn(2,4,8,requires_grad=True)
        a = torch.ones(2,4,4)
        self.assertTrue(torch.equal(layer(h,a,a),h))
        layer(h,a,a).square().sum().backward()
        self.assertIsNotNone(layer.local_gate.grad)
        self.assertIsNotNone(layer.component_gate.grad)

    def test_classifier_and_predictor(self):
        image = torch.randn(2,3,64,64)
        predictor = MaskPredictor().eval()
        with torch.no_grad():
            probability = predictor(image[:,:1]).sigmoid().squeeze(1)
        self.assertEqual(tuple(probability.shape),(2,64,64))
        for k in (8,16):
            model = MaskTopo(token_count=k).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()),149123)
            topology = build_topology(probability,token_count=k)
            with torch.no_grad(): result=model(image,*topology.tensors())
            self.assertEqual(tuple(result.shape),(2,))
            self.assertTrue(torch.isfinite(result).all())

    def test_invalid_input(self):
        for kwargs in [{'token_count':0},{'token_count':257},{'closing_iterations':3},{'threshold':-1}]:
            with self.assertRaises(ValueError):build_topology(np.zeros((64,64)),**kwargs)
        with self.assertRaises(ValueError):build_topology(np.zeros((32,32)))
        with self.assertRaises(ValueError):build_topology(np.full((64,64),np.nan))

if __name__ == '__main__':unittest.main()
