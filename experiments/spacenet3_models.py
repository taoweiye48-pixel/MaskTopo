from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label

from crackforest_real_gate import coarse_graph8, patch_component_ids
from ph_token_baselines import DESCRIPTOR_DIM, ph_descriptors
from topobridge_mvp import PatchStem
from topocoarsen_oracle import oracle_assignment


MODEL_NAMES = (
    "grid",
    "tokenlearner",
    "perceiver_resampler",
    "tome_style",
    "mask_guided_queries",
    "ph_only",
    "ph_guided",
    "mask_assignment_identity",
    "shuffled_topology",
    "mask_topo",
)


def fine_coordinates() -> torch.Tensor:
    axis = (torch.arange(16, dtype=torch.float32) + 0.5) / 16
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((2 * rows - 1, 2 * columns - 1), dim=-1).reshape(-1, 2)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.GELU(), nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.GELU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class RoadUNet(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(3, base)
        self.encoder2 = ConvBlock(base, 2 * base)
        self.encoder3 = ConvBlock(2 * base, 4 * base)
        self.bottleneck = ConvBlock(4 * base, 8 * base)
        self.decoder3 = ConvBlock(12 * base, 4 * base)
        self.decoder2 = ConvBlock(6 * base, 2 * base)
        self.decoder1 = ConvBlock(3 * base, base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(image)
        second = self.encoder2(F.max_pool2d(first, 2))
        third = self.encoder3(F.max_pool2d(second, 2))
        bottleneck = self.bottleneck(F.max_pool2d(third, 2))
        value = self.decoder3(torch.cat((F.interpolate(bottleneck, size=third.shape[-2:], mode="bilinear", align_corners=False), third), dim=1))
        value = self.decoder2(torch.cat((F.interpolate(value, size=second.shape[-2:], mode="bilinear", align_corners=False), second), dim=1))
        value = self.decoder1(torch.cat((F.interpolate(value, size=first.shape[-2:], mode="bilinear", align_corners=False), first), dim=1))
        return self.output(value).squeeze(1)


def balanced_binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_bool = target > 0.5
    parts = []
    if torch.any(target_bool):
        parts.append(F.binary_cross_entropy_with_logits(logits[target_bool], torch.ones_like(logits[target_bool])))
    if torch.any(~target_bool):
        parts.append(F.binary_cross_entropy_with_logits(logits[~target_bool], torch.zeros_like(logits[~target_bool])))
    return torch.stack(parts).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    bce = balanced_binary_loss(logits, target)
    probability = torch.sigmoid(logits)
    intersection = torch.sum(probability * target, dim=(1, 2))
    denominator = torch.sum(probability + target, dim=(1, 2))
    dice = torch.mean(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))
    return bce + dice, {"balanced_bce": float(bce.detach()), "dice_loss": float(dice.detach())}


class GraphMixLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.local_message = nn.Linear(dim, dim)
        self.component_message = nn.Linear(dim, dim)
        self.local_gate = nn.Parameter(torch.zeros(()))
        self.component_gate = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor, reachability: torch.Tensor) -> torch.Tensor:
        local_weights = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1)
        component_weights = reachability / reachability.sum(dim=-1, keepdim=True).clamp_min(1)
        local = torch.bmm(local_weights, tokens)
        component = torch.bmm(component_weights, tokens)
        return tokens + torch.tanh(self.local_gate) * self.local_message(local) + torch.tanh(self.component_gate) * self.component_message(component)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.feed_forward = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * dim, dim))

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        update, _ = self.attention(self.query_norm(queries), self.token_norm(tokens), self.token_norm(tokens), need_weights=False)
        queries = queries + update
        return queries + self.feed_forward(queries)


class SpatialGraphDecoder(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.token_count = token_count
        self.metadata = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.project = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.graph_mix = GraphMixLayer(dim)
        self.query_position = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query_embedding = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        self.blocks = nn.ModuleList([DecoderBlock(dim) for _ in range(2)])
        self.upsample = nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1), nn.BatchNorm2d(dim // 2), nn.GELU(), nn.ConvTranspose2d(dim // 2, dim // 4, 4, stride=2, padding=1), nn.BatchNorm2d(dim // 4), nn.GELU(), nn.Conv2d(dim // 4, 1, 3, padding=1))
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(self, tokens: torch.Tensor, metadata: torch.Tensor, adjacency: torch.Tensor, reachability: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.token_count:
            raise ValueError(f"Expected {self.token_count} tokens, got {tokens.shape[1]}")
        tokens = self.project(tokens) + self.metadata(metadata)
        tokens = self.graph_mix(tokens, adjacency.to(tokens.dtype), reachability.to(tokens.dtype))
        coordinates = self.coordinates.to(tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        queries = self.query_embedding.expand(tokens.shape[0], -1, -1) + self.query_position(coordinates)
        for block in self.blocks:
            queries = block(queries, tokens)
        return self.upsample(queries.transpose(1, 2).reshape(tokens.shape[0], -1, 16, 16)).squeeze(1)


class TokenModel(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.dim = dim
        self.token_count = token_count
        self.decoder = SpatialGraphDecoder(dim, token_count)

    def identity_graph(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        identity = torch.eye(self.token_count, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return identity, identity

    def decode_identity(self, tokens: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        adjacency, reachability = self.identity_graph(tokens.shape[0], tokens.device, tokens.dtype)
        return self.decoder(tokens, metadata, adjacency, reachability)


def pool_assignment(features: torch.Tensor, assignment: torch.Tensor, token_count: int, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = features.flatten(2).transpose(1, 2)
    one_hot = F.one_hot(assignment.reshape(assignment.shape[0], -1), num_classes=token_count).to(flat.dtype)
    counts = one_hot.sum(dim=1).clamp_min(1)
    tokens = torch.bmm(one_hot.transpose(1, 2), flat) / counts.unsqueeze(-1)
    coordinate_bank = coordinates.to(flat.dtype).unsqueeze(0).expand(flat.shape[0], -1, -1)
    centroids = torch.bmm(one_hot.transpose(1, 2), coordinate_bank) / counts.unsqueeze(-1)
    mass = (counts / 256.0).unsqueeze(-1)
    return tokens, torch.cat((centroids, mass), dim=-1)


class GridModel(TokenModel):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        if token_count != 16:
            raise ValueError("Frozen grid model requires K=16")
        self.stem = PatchStem(dim)
        assignment = torch.arange(16, dtype=torch.long).reshape(4, 4).repeat_interleave(4, 0).repeat_interleave(4, 1)
        self.register_buffer("assignment", assignment, persistent=False)
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(self, image: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        assignment = self.assignment.unsqueeze(0).expand(image.shape[0], -1, -1)
        tokens, metadata = pool_assignment(self.stem(image), assignment, self.token_count, self.coordinates)
        return self.decode_identity(tokens, metadata)


class TokenLearnerModel(TokenModel):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.score = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, token_count, 1))
        self.value = nn.Conv2d(dim, dim, 1)
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(self, image: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        features = self.stem(image)
        values = self.value(features).flatten(2).transpose(1, 2)
        raw = torch.sigmoid(self.score(features)).flatten(2)
        mass_raw = raw.sum(dim=-1).clamp_min(1e-6)
        weights = raw / mass_raw.unsqueeze(-1)
        tokens = torch.bmm(weights, values)
        coordinates = self.coordinates.to(values.dtype).unsqueeze(0).expand(image.shape[0], -1, -1)
        centroids = torch.bmm(weights, coordinates)
        mass = (mass_raw / mass_raw.sum(dim=-1, keepdim=True)).unsqueeze(-1)
        return self.decode_identity(tokens, torch.cat((centroids, mass), dim=-1))


class ResamplerBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.feed_forward = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * dim, dim))

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        update, weights = self.attention(self.query_norm(latents), self.context_norm(context), self.context_norm(context), need_weights=True, average_attn_weights=True)
        latents = latents + update
        return latents + self.feed_forward(latents), weights


class PerceiverModel(TokenModel):
    def __init__(self, dim: int, token_count: int, mask_guided: bool) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.queries = nn.Parameter(torch.randn(token_count, dim) * 0.02)
        self.position = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mask_guided = mask_guided
        self.mask_embedding = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim)) if mask_guided else None
        self.blocks = nn.ModuleList([ResamplerBlock(dim) for _ in range(2)])
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(self, image: torch.Tensor, mask_probability: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        context = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(context.dtype).unsqueeze(0).expand(image.shape[0], -1, -1)
        context = context + self.position(coordinates)
        if self.mask_guided:
            if self.mask_embedding is None:
                raise AssertionError("Missing mask embedding")
            mask_tokens = F.adaptive_avg_pool2d(mask_probability.unsqueeze(1), (16, 16)).flatten(2).transpose(1, 2)
            context = context + self.mask_embedding(mask_tokens)
        latents = self.queries.unsqueeze(0).expand(image.shape[0], -1, -1)
        weights = None
        for block in self.blocks:
            latents, weights = block(latents, context)
        if weights is None:
            raise AssertionError("Missing resampler weights")
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        centroids = torch.bmm(weights, coordinates)
        mass = torch.full((image.shape[0], self.token_count, 1), 1.0 / self.token_count, device=image.device, dtype=latents.dtype)
        return self.decode_identity(latents, torch.cat((centroids, mass), dim=-1))


class ToMeModel(TokenModel):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        if token_count != 16:
            raise ValueError("Frozen ToMe-style model requires K=16")
        self.stem = PatchStem(dim)
        self.position = nn.Linear(2, dim)
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def merge_once(self, tokens: torch.Tensor, coordinates: torch.Tensor, mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = tokens[:, 0::2], tokens[:, 1::2]
        left_coordinates, right_coordinates = coordinates[:, 0::2], coordinates[:, 1::2]
        left_mass, right_mass = mass[:, 0::2], mass[:, 1::2]
        left_key = F.normalize(left + self.position(left_coordinates), dim=-1)
        right_key = F.normalize(right + self.position(right_coordinates), dim=-1)
        log_pairing = torch.bmm(left_key, right_key.transpose(1, 2)) / 0.10
        for _ in range(3):
            log_pairing = log_pairing - torch.logsumexp(log_pairing, dim=-1, keepdim=True)
            log_pairing = log_pairing - torch.logsumexp(log_pairing, dim=-2, keepdim=True)
        pairing = torch.exp(log_pairing)
        transported_mass = torch.bmm(pairing.transpose(1, 2), left_mass)
        new_mass = (right_mass + transported_mass).clamp_min(1e-6)
        new_tokens = (right * right_mass + torch.bmm(pairing.transpose(1, 2), left * left_mass)) / new_mass
        new_coordinates = (right_coordinates * right_mass + torch.bmm(pairing.transpose(1, 2), left_coordinates * left_mass)) / new_mass
        return new_tokens, new_coordinates, new_mass

    def forward(self, image: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(tokens.dtype).unsqueeze(0).expand(image.shape[0], -1, -1)
        mass = torch.full((image.shape[0], 256, 1), 1.0 / 256, device=image.device, dtype=tokens.dtype)
        for _ in range(4):
            tokens, coordinates, mass = self.merge_once(tokens, coordinates, mass)
        mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self.decode_identity(tokens, torch.cat((coordinates, mass), dim=-1))


class PHOnlyModel(TokenModel):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        self.descriptor = nn.Sequential(nn.Linear(DESCRIPTOR_DIM, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, ph_descriptor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        tokens = self.descriptor(ph_descriptor)
        persistence = ph_descriptor[..., 2:3] * ph_descriptor[..., 6:7]
        mass = persistence / persistence.sum(dim=1, keepdim=True).clamp_min(1e-6)
        metadata = torch.cat((ph_descriptor[..., 7:9], mass), dim=-1)
        return self.decode_identity(tokens, metadata)


class PHGuidedModel(TokenModel):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.descriptor = nn.Sequential(nn.Linear(DESCRIPTOR_DIM, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.position = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([ResamplerBlock(dim) for _ in range(2)])
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(self, image: torch.Tensor, ph_descriptor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        context = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(context.dtype).unsqueeze(0).expand(image.shape[0], -1, -1)
        context = context + self.position(coordinates)
        latents = self.descriptor(ph_descriptor)
        weights = None
        for block in self.blocks:
            latents, weights = block(latents, context)
        if weights is None:
            raise AssertionError("Missing PH-guided weights")
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        centroids = torch.bmm(weights, coordinates)
        persistence = ph_descriptor[..., 2:3] * ph_descriptor[..., 6:7]
        mass = persistence / persistence.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self.decode_identity(latents, torch.cat((centroids, mass), dim=-1))


class ArtifactModel(TokenModel):
    def __init__(self, dim: int, token_count: int, mode: str) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.mode = mode
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(self, image: torch.Tensor, assignment: torch.Tensor, adjacency: torch.Tensor, reachability: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        tokens, metadata = pool_assignment(self.stem(image), assignment.long(), self.token_count, self.coordinates)
        if self.mode == "mask_assignment_identity":
            adjacency, reachability = self.identity_graph(tokens.shape[0], tokens.device, tokens.dtype)
        return self.decoder(tokens, metadata, adjacency, reachability)


def make_model(name: str, dim: int = 64, token_count: int = 16) -> nn.Module:
    if name == "grid":
        return GridModel(dim, token_count)
    if name == "tokenlearner":
        return TokenLearnerModel(dim, token_count)
    if name == "perceiver_resampler":
        return PerceiverModel(dim, token_count, False)
    if name == "mask_guided_queries":
        return PerceiverModel(dim, token_count, True)
    if name == "tome_style":
        return ToMeModel(dim, token_count)
    if name == "ph_only":
        return PHOnlyModel(dim, token_count)
    if name == "ph_guided":
        return PHGuidedModel(dim, token_count)
    if name in {"mask_assignment_identity", "shuffled_topology", "mask_topo"}:
        return ArtifactModel(dim, token_count, name)
    raise ValueError(name)


def limit_groups(groups: np.ndarray, maximum_foreground: int = 15) -> np.ndarray:
    groups = groups.astype(np.int16, copy=True)
    components = [int(item) for item in np.unique(groups) if int(item) > 0]
    if len(components) > maximum_foreground:
        sizes = {component: int(np.sum(groups == component)) for component in components}
        keep = set(sorted(components, key=sizes.__getitem__, reverse=True)[:maximum_foreground])
        for component in components:
            if component not in keep:
                groups[groups == component] = 0
    ordered = [0, *sorted(int(item) for item in np.unique(groups) if item > 0)] if np.any(groups == 0) else sorted(int(item) for item in np.unique(groups))
    remap = {old: new for new, old in enumerate(ordered)}
    return np.vectorize(remap.__getitem__, otypes=[np.int16])(groups)


def build_artifacts(probabilities: np.ndarray, token_count: int = 16, threshold: float = 0.5, shuffle_seed: int = 20260731) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    count = probabilities.shape[0]
    outputs = {name: {"assignment": np.empty((count, 16, 16), dtype=np.int16), "adjacency": np.empty((count, token_count, token_count), dtype=np.uint8), "reachability": np.empty((count, token_count, token_count), dtype=np.uint8)} for name in ("mask_assignment_identity", "shuffled_topology", "mask_topo")}
    rng = np.random.default_rng(shuffle_seed)
    component_counts = []
    for index, probability in enumerate(probabilities):
        component_map, component_count = label(probability >= threshold, structure=np.ones((3, 3), dtype=np.uint8))
        groups = limit_groups(patch_component_ids(component_map), token_count - 1)
        component_counts.append(int(component_count))
        shuffled = groups.reshape(-1)[rng.permutation(groups.size)].reshape(16, 16)
        for name, current_groups in (("mask_assignment_identity", groups), ("shuffled_topology", shuffled), ("mask_topo", groups)):
            assignment = oracle_assignment(current_groups, token_count, False)
            if name == "mask_assignment_identity":
                graph = closure = np.eye(token_count, dtype=np.uint8)
            else:
                graph, closure = coarse_graph8(assignment, current_groups, token_count, topology_aware=True)
            outputs[name]["assignment"][index] = assignment
            outputs[name]["adjacency"][index] = graph
            outputs[name]["reachability"][index] = closure
    return outputs, {"threshold": threshold, "closing": 0, "shuffle_seed": shuffle_seed, "component_count_mean_64": float(np.mean(component_counts)), "component_count_p95_64": float(np.quantile(component_counts, 0.95)), "all_tokens_nonempty": bool(all(np.unique(outputs[name]["assignment"][index]).size == token_count for name in outputs for index in range(count)))}


def build_ph(probabilities: np.ndarray, token_count: int = 16) -> tuple[np.ndarray, dict[str, float]]:
    descriptors = np.empty((probabilities.shape[0], token_count, DESCRIPTOR_DIM), dtype=np.float32)
    available = []
    for index, probability in enumerate(probabilities):
        descriptors[index], diagnostics = ph_descriptors(probability, token_count)
        available.append(int(diagnostics["available_pairs"]))
        if (index + 1) % 250 == 0 or index + 1 == probabilities.shape[0]:
            print(f"PH_BUILD progress={index + 1}/{probabilities.shape[0]}", flush=True)
    valid = descriptors[..., 6] > 0.5
    return descriptors, {"available_pairs_mean": float(np.mean(available)), "padding_rate": float(1.0 - valid.mean()), "selected_h0_mean": float(((descriptors[..., 3] > 0.5) & valid).sum(axis=1).mean()), "selected_h1_mean": float(((descriptors[..., 4] > 0.5) & valid).sum(axis=1).mean())}


def run_self_test() -> None:
    torch.manual_seed(7)
    image = torch.rand(2, 3, 64, 64)
    probabilities = np.zeros((2, 64, 64), dtype=np.float32)
    probabilities[:, 30:34, 5:59] = 0.9
    artifacts, diagnostics = build_artifacts(probabilities)
    assert diagnostics["all_tokens_nonempty"]
    descriptors, _ = build_ph(probabilities)
    common = {"image": image, "mask_probability": torch.from_numpy(probabilities), "ph_descriptor": torch.from_numpy(descriptors)}
    for name in MODEL_NAMES:
        model = make_model(name, dim=32, token_count=16).eval()
        inputs = dict(common)
        if name in artifacts:
            inputs.update({key: torch.from_numpy(value) for key, value in artifacts[name].items()})
        with torch.no_grad():
            logits = model(**inputs)
        assert logits.shape == (2, 64, 64), (name, logits.shape)
        loss, _ = segmentation_loss(logits, torch.zeros_like(logits))
        assert torch.isfinite(loss), name
    predictor = RoadUNet(base=8)
    assert predictor(image).shape == (2, 64, 64)
    print("SPACENET3_MODELS_SELF_TEST_PASS")


if __name__ == "__main__":
    run_self_test()
