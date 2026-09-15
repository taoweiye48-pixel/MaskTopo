from __future__ import annotations

import hashlib
import inspect
import json
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from torchvision.models import ResNet34_Weights, resnet34

from crackforest_real_gate import coarse_graph8, patch_component_ids
from spacenet3_models import GraphMixLayer, limit_groups
from topocoarsen_oracle import oracle_assignment


METHODS = ("skip_only", "grid", "mask_assignment_identity", "shuffled_topology", "mask_topo")
TOKEN_COUNT = 16
DIMENSION = 64
IMAGENET_WEIGHTS = ResNet34_Weights.IMAGENET1K_V1
IMAGENET_MEAN = tuple(float(value) for value in IMAGENET_WEIGHTS.transforms().mean)
IMAGENET_STD = tuple(float(value) for value in IMAGENET_WEIGHTS.transforms().std)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def function_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet34(weights=IMAGENET_WEIGHTS if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.decode4 = ConvBlock(512 + 256, 256)
        self.decode3 = ConvBlock(256 + 128, 128)
        self.decode2 = ConvBlock(128 + 64, 64)
        self.decode1 = ConvBlock(64 + 64, 32)
        self.decode0 = ConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stem = self.stem(image)
        layer1 = self.layer1(self.maxpool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        value = self.decode4(torch.cat((F.interpolate(layer4, size=layer3.shape[-2:], mode="bilinear", align_corners=False), layer3), dim=1))
        value = self.decode3(torch.cat((F.interpolate(value, size=layer2.shape[-2:], mode="bilinear", align_corners=False), layer2), dim=1))
        value = self.decode2(torch.cat((F.interpolate(value, size=layer1.shape[-2:], mode="bilinear", align_corners=False), layer1), dim=1))
        value = self.decode1(torch.cat((F.interpolate(value, size=stem.shape[-2:], mode="bilinear", align_corners=False), stem), dim=1))
        value = self.decode0(F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False))
        return self.output(value).squeeze(1)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = torch.sum(probability * target, dim=(1, 2))
    denominator = torch.sum(probability + target, dim=(1, 2))
    return torch.mean(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-image, 3, stride=1, padding=1)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeleton(image: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits).unsqueeze(1)
    target = target.unsqueeze(1)
    prediction_skeleton = soft_skeleton(probability)
    target_skeleton = soft_skeleton(target)
    topology_precision = (prediction_skeleton * target).sum(dim=(1, 2, 3)) / prediction_skeleton.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    topology_sensitivity = (target_skeleton * probability).sum(dim=(1, 2, 3)) / target_skeleton.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return torch.mean(1.0 - (2.0 * topology_precision * topology_sensitivity + 1e-6) / (topology_precision + topology_sensitivity + 1e-6))


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(4.0, device=logits.device))
    dice = soft_dice_loss(logits, target)
    topology = soft_cldice_loss(logits, target)
    total = bce + dice + 0.5 * topology
    return total, {"bce_pos4": float(bce.detach()), "soft_dice": float(dice.detach()), "soft_cldice": float(topology.detach())}


class Stage1SpatialEncoder(nn.Module):
    def __init__(self, dim: int = DIMENSION) -> None:
        super().__init__()
        self.s4 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.to_s8 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.BatchNorm2d(dim), nn.GELU())
        self.to_t16 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.BatchNorm2d(dim), nn.GELU())

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s4 = self.s4(image)
        t16 = self.to_t16(self.to_s8(s4))
        return s4, t16


def fine_coordinates() -> torch.Tensor:
    axis = (torch.arange(16, dtype=torch.float32) + 0.5) / 16
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((2 * rows - 1, 2 * columns - 1), dim=-1).reshape(-1, 2)


def pool_assignment(features: torch.Tensor, assignment: torch.Tensor, token_count: int, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = features.flatten(2).transpose(1, 2)
    one_hot = F.one_hot(assignment.reshape(assignment.shape[0], -1), num_classes=token_count).to(flat.dtype)
    counts = one_hot.sum(dim=1).clamp_min(1)
    tokens = torch.bmm(one_hot.transpose(1, 2), flat) / counts.unsqueeze(-1)
    coordinate_bank = coordinates.to(flat.dtype).unsqueeze(0).expand(flat.shape[0], -1, -1)
    centroids = torch.bmm(one_hot.transpose(1, 2), coordinate_bank) / counts.unsqueeze(-1)
    mass = (counts / 256.0).unsqueeze(-1)
    return tokens, torch.cat((centroids, mass), dim=-1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0.0, batch_first=True)
        self.feed_forward = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor, zero_update: bool) -> tuple[torch.Tensor, torch.Tensor]:
        update, _ = self.attention(self.query_norm(queries), self.token_norm(tokens), self.token_norm(tokens), need_weights=False)
        applied = update * 0.0 if zero_update else update
        queries = queries + applied
        return queries + self.feed_forward(queries), applied


class Stage1Decoder(nn.Module):
    def __init__(self, dim: int = DIMENSION, token_count: int = TOKEN_COUNT) -> None:
        super().__init__()
        self.token_count = token_count
        self.token_project = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.metadata = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.graph_mix = GraphMixLayer(dim)
        self.blocks = nn.ModuleList([CrossAttentionBlock(dim) for _ in range(2)])
        self.upsample1 = ConvBlock(dim, 32)
        self.upsample2 = ConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, 1)
        self.last_cross_update_absmax = float("nan")

    def forward(
        self,
        s4: torch.Tensor,
        tokens: torch.Tensor,
        metadata: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
        zero_update: bool,
    ) -> torch.Tensor:
        tokens = self.token_project(tokens) + self.metadata(metadata)
        tokens = self.graph_mix(tokens, adjacency.to(tokens.dtype), reachability.to(tokens.dtype))
        queries = s4.flatten(2).transpose(1, 2)
        updates: list[torch.Tensor] = []
        for block in self.blocks:
            queries, update = block(queries, tokens, zero_update)
            updates.append(update)
        self.last_cross_update_absmax = max(float(update.detach().abs().max()) for update in updates)
        value = queries.transpose(1, 2).reshape(s4.shape)
        value = self.upsample1(F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False))
        value = self.upsample2(F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False))
        return self.output(value).squeeze(1)


class TokenConditionedRoadModel(nn.Module):
    def __init__(self, mode: str, dim: int = DIMENSION, token_count: int = TOKEN_COUNT) -> None:
        super().__init__()
        if mode not in METHODS:
            raise ValueError(mode)
        if token_count != 16:
            raise ValueError("Stage 1 is frozen at K=16")
        self.mode = mode
        self.token_count = token_count
        self.encoder = Stage1SpatialEncoder(dim)
        self.decoder = Stage1Decoder(dim, token_count)
        grid = torch.arange(16, dtype=torch.long).reshape(4, 4).repeat_interleave(4, 0).repeat_interleave(4, 1)
        self.register_buffer("grid_assignment", grid, persistent=False)
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(
        self,
        image: torch.Tensor,
        assignment: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
        reachability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s4, t16 = self.encoder(image)
        batch_size = image.shape[0]
        if self.mode in {"skip_only", "grid"}:
            assignment = self.grid_assignment.unsqueeze(0).expand(batch_size, -1, -1)
        elif assignment is None:
            raise ValueError(f"{self.mode} requires an assignment")
        tokens, metadata = pool_assignment(t16, assignment.long(), self.token_count, self.coordinates)
        if self.mode in {"skip_only", "grid", "mask_assignment_identity"}:
            identity = torch.eye(self.token_count, device=image.device, dtype=tokens.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            adjacency, reachability = identity, identity
        elif adjacency is None or reachability is None:
            raise ValueError(f"{self.mode} requires graph matrices")
        return self.decoder(s4, tokens, metadata, adjacency, reachability, zero_update=self.mode == "skip_only")


def make_model(name: str, dim: int = DIMENSION, token_count: int = TOKEN_COUNT) -> TokenConditionedRoadModel:
    return TokenConditionedRoadModel(name, dim=dim, token_count=token_count)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def structure_hash(model: nn.Module) -> str:
    structure = {
        "modules": [(name, module.__class__.__name__) for name, module in model.named_modules()],
        "parameters": [(name, list(parameter.shape)) for name, parameter in model.named_parameters()],
    }
    return hashlib.sha256(json.dumps(structure, sort_keys=True).encode("utf-8")).hexdigest()


def imagenet_weight_record() -> dict[str, Any]:
    model = ResNet34UNet(pretrained=True)
    del model
    filename = Path(urllib.parse.urlparse(IMAGENET_WEIGHTS.url).path).name
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not weight_path.is_file():
        raise FileNotFoundError(f"torchvision did not cache the requested weight: {weight_path}")
    return {
        "enum": "torchvision.models.ResNet34_Weights.IMAGENET1K_V1",
        "url": IMAGENET_WEIGHTS.url,
        "path": str(weight_path.resolve()),
        "sha256": sha256_file(weight_path),
        "bytes": weight_path.stat().st_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }


def build_artifacts(
    probabilities: np.ndarray,
    threshold: float,
    token_count: int = TOKEN_COUNT,
    shuffle_seed: int = 20260850,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if threshold not in tuple(round(index * 0.05, 2) for index in range(1, 20)):
        raise ValueError(f"Threshold is outside the frozen candidate set: {threshold}")
    count = int(probabilities.shape[0])
    assignment_bank = np.empty((count, 16, 16), dtype=np.int16)
    adjacency_identity = np.repeat(np.eye(token_count, dtype=np.uint8)[None], count, axis=0)
    reachability_identity = adjacency_identity.copy()
    adjacency_real = np.empty((count, token_count, token_count), dtype=np.uint8)
    reachability_real = np.empty_like(adjacency_real)
    adjacency_shuffled = np.empty_like(adjacency_real)
    reachability_shuffled = np.empty_like(adjacency_real)
    uninformative = np.zeros(count, dtype=np.uint8)
    dropped_component_count = np.empty(count, dtype=np.int16)
    dropped_pixel_fraction = np.empty(count, dtype=np.float32)
    permutations = np.empty((count, token_count), dtype=np.int16)
    rng = np.random.default_rng(shuffle_seed)
    for index, probability in enumerate(probabilities):
        component_map, component_count = label(probability >= threshold, structure=np.ones((3, 3), dtype=np.uint8))
        raw_groups = patch_component_ids(component_map)
        represented = [int(item) for item in np.unique(raw_groups) if int(item) > 0]
        patch_sizes = {item: int(np.sum(raw_groups == item)) for item in represented}
        keep = set(sorted(represented, key=lambda item: (patch_sizes[item], -item), reverse=True)[: token_count - 1])
        dropped = set(range(1, int(component_count) + 1)) - keep
        foreground_pixels = int((component_map > 0).sum())
        dropped_component_count[index] = len(dropped)
        dropped_pixel_fraction[index] = float(np.isin(component_map, list(dropped)).sum() / foreground_pixels) if foreground_pixels and dropped else 0.0
        groups = limit_groups(raw_groups, token_count - 1)
        assignment = oracle_assignment(groups, token_count, False)
        graph, closure = coarse_graph8(assignment, groups, token_count, topology_aware=True)
        chosen = np.arange(token_count)
        changed = False
        for _ in range(128):
            candidate = rng.permutation(token_count)
            if np.array_equal(candidate, np.arange(token_count)):
                continue
            shuffled_graph = graph[np.ix_(candidate, candidate)]
            shuffled_closure = closure[np.ix_(candidate, candidate)]
            if not np.array_equal(shuffled_graph, graph) or not np.array_equal(shuffled_closure, closure):
                chosen = candidate
                changed = True
                break
        if changed:
            adjacency_shuffled[index] = graph[np.ix_(chosen, chosen)]
            reachability_shuffled[index] = closure[np.ix_(chosen, chosen)]
        else:
            adjacency_shuffled[index] = graph
            reachability_shuffled[index] = closure
            uninformative[index] = 1
        assignment_bank[index] = assignment
        adjacency_real[index] = graph
        reachability_real[index] = closure
        permutations[index] = chosen
    arrays = {
        "assignment": assignment_bank,
        "adjacency_identity": adjacency_identity,
        "reachability_identity": reachability_identity,
        "adjacency_real": adjacency_real,
        "reachability_real": reachability_real,
        "adjacency_shuffled": adjacency_shuffled,
        "reachability_shuffled": reachability_shuffled,
        "shuffle_permutation": permutations,
        "shuffled_uninformative": uninformative,
        "dropped_component_count": dropped_component_count,
        "dropped_pixel_fraction": dropped_pixel_fraction,
    }
    metadata = {
        "threshold": threshold,
        "token_count": token_count,
        "shuffle_seed": shuffle_seed,
        "assignment_shared_across_mask_artifact_methods": True,
        "token_and_metadata_construction_shared_across_mask_artifact_methods": True,
        "shuffled_uninformative_count": int(uninformative.sum()),
        "shuffled_uninformative_fraction": float(uninformative.mean()),
        "dropped_component_count_mean": float(dropped_component_count.mean()),
        "dropped_pixel_fraction_mean": float(dropped_pixel_fraction.mean()),
    }
    return arrays, metadata


def artifact_inputs(name: str, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if name in {"skip_only", "grid"}:
        return {}
    if name == "mask_assignment_identity":
        return {
            "assignment": batch["assignment"],
            "adjacency": batch["adjacency_identity"],
            "reachability": batch["reachability_identity"],
        }
    if name == "shuffled_topology":
        return {
            "assignment": batch["assignment"],
            "adjacency": batch["adjacency_shuffled"],
            "reachability": batch["reachability_shuffled"],
        }
    if name == "mask_topo":
        return {
            "assignment": batch["assignment"],
            "adjacency": batch["adjacency_real"],
            "reachability": batch["reachability_real"],
        }
    raise ValueError(name)


def run_self_test() -> None:
    torch.manual_seed(7)
    predictor = ResNet34UNet(pretrained=False).eval()
    image = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        logits = predictor(image)
    assert logits.shape == (1, 256, 256)
    loss, _ = segmentation_loss(logits, torch.zeros_like(logits))
    assert torch.isfinite(loss)

    probabilities = np.zeros((2, 256, 256), dtype=np.float32)
    probabilities[:, 120:136, 20:236] = 0.9
    probabilities[1, 20:236, 124:132] = 0.9
    artifacts, metadata = build_artifacts(probabilities, threshold=0.5, shuffle_seed=17)
    assert metadata["assignment_shared_across_mask_artifact_methods"]
    for key in ("adjacency_real", "reachability_real", "adjacency_shuffled", "reachability_shuffled"):
        assert np.array_equal(artifacts[key], artifacts[key].transpose(0, 2, 1)), key
    informative = artifacts["shuffled_uninformative"] == 0
    if informative.any():
        assert np.any(artifacts["adjacency_real"][informative] != artifacts["adjacency_shuffled"][informative])
    batch = {key: torch.from_numpy(value) for key, value in artifacts.items() if value.dtype != np.float32}
    batch["dropped_pixel_fraction"] = torch.from_numpy(artifacts["dropped_pixel_fraction"])
    model_image = torch.rand(2, 3, 256, 256)
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    updates: dict[str, float] = {}
    shapes: dict[str, Any] = {}
    with torch.no_grad():
        for name in METHODS:
            model = make_model(name, dim=32).eval()
            output = model(model_image, **artifact_inputs(name, batch))
            assert output.shape == (2, 256, 256), (name, output.shape)
            counts[name] = parameter_count(model)
            hashes[name] = structure_hash(model)
            updates[name] = model.decoder.last_cross_update_absmax
            s4, t16 = model.encoder(model_image)
            shapes[name] = {"S4": list(s4.shape), "T16": list(t16.shape), "skip_elements_per_sample": int(s4[0].numel())}
    assert len(set(counts.values())) == 1 and len(set(hashes.values())) == 1
    assert updates["skip_only"] == 0.0
    assert all(updates[name] > 0 for name in METHODS if name != "skip_only")
    assert len({json.dumps(value, sort_keys=True) for value in shapes.values()}) == 1
    print(json.dumps({"status": "SPACENET3_STAGE1_MODELS_SELF_TEST_PASS", "parameters": counts, "structure_hashes": hashes, "cross_attention_update_absmax": updates, "skip_shapes": shapes, "artifact_metadata": metadata}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run_self_test()
