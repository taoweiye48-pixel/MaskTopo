"""MaskTopo: component-aligned token coarsening for connectivity."""
from .model import MaskTopo, MaskPredictor, GatedTopologicalMessaging
from .coarsening import Topology, build_topology
__all__ = ["MaskTopo", "MaskPredictor", "GatedTopologicalMessaging", "Topology", "build_topology"]
__version__ = "0.1.0"
