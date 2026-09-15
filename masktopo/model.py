"""MaskTopo classifier and image-only structure predictor.

Neural layers are extracted unchanged from the experiment implementation.
MaskTopo keeps the historical checkpoint key schema and parameter count.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class PatchStem(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class ConvConnector(nn.Module):
    def __init__(self, dim: int, side: int) -> None:
        super().__init__()
        self.side = side
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        local = features + self.local(features)
        return F.adaptive_avg_pool2d(local, (self.side, self.side)), {}

def fine_coordinates() -> torch.Tensor:
    axis = (torch.arange(16, dtype=torch.float32) + 0.5) / 16
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((2 * rows - 1, 2 * cols - 1), dim=-1).reshape(-1, 2)


class GraphMixLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.local_message = nn.Linear(dim, dim)
        self.component_message = nn.Linear(dim, dim)
        self.local_gate = nn.Parameter(torch.zeros(()))
        self.component_gate = nn.Parameter(torch.zeros(()))
        # Forward-only controls used by parameter-matched branch ablations.
        # They are ordinary Python attributes, so the historical state-dict
        # schema and parameter count remain unchanged.
        self.use_local = True
        self.use_reachability = True

    def forward(
        self,
        tokens: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        local_weights = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1)
        global_weights = reachability / reachability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        mixed = tokens
        if self.use_local:
            local = torch.bmm(local_weights, tokens)
            mixed = mixed + torch.tanh(self.local_gate) * self.local_message(local)
        if self.use_reachability:
            component = torch.bmm(global_weights, tokens)
            mixed = mixed + (
                torch.tanh(self.component_gate)
                * self.component_message(component)
            )
        return mixed


class CoarseGraphHead(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.metadata = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.graph_layers = nn.ModuleList([GraphMixLayer(dim)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=4 * dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 1)
        self.token_count = token_count

    def forward(
        self,
        tokens: torch.Tensor,
        metadata: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens + self.metadata(metadata)
        for layer in self.graph_layers:
            tokens = layer(tokens, adjacency, reachability)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1))
        return self.classifier(self.norm(encoded[:, 0])).squeeze(-1)


class TopoCoarsenModel(nn.Module):
    def __init__(self, name: str, dim: int, token_count: int) -> None:
        super().__init__()
        self.name = name
        self.token_count = token_count
        self.stem = PatchStem(dim)
        side = int(math.isqrt(token_count))
        self.conv = ConvConnector(dim, side) if name == "conv" else None
        self.project = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.head = CoarseGraphHead(dim, token_count)
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def pool(
        self, features: torch.Tensor, assignment: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = features.flatten(2).transpose(1, 2)
        one_hot = F.one_hot(
            assignment.reshape(assignment.shape[0], -1),
            num_classes=self.token_count,
        ).to(flat.dtype)
        counts = one_hot.sum(dim=1).clamp_min(1)
        tokens = torch.bmm(one_hot.transpose(1, 2), flat) / counts.unsqueeze(-1)
        coordinate_bank = self.coordinates.unsqueeze(0).expand(
            features.shape[0], -1, -1
        )
        centroids = (
            torch.bmm(one_hot.transpose(1, 2), coordinate_bank)
            / counts.unsqueeze(-1)
        )
        mass = (counts / 256.0).unsqueeze(-1)
        return tokens, torch.cat((centroids, mass), dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        assignment: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        features = self.stem(image)
        if self.name == "conv":
            if self.conv is None:
                raise AssertionError("Missing convolutional connector.")
            compressed, _ = self.conv(features)
            tokens = compressed.flatten(2).transpose(1, 2)
            side = int(math.isqrt(self.token_count))
            axis = (torch.arange(side, device=image.device, dtype=tokens.dtype) + 0.5) / side
            rows, cols = torch.meshgrid(axis, axis, indexing="ij")
            centroids = torch.stack(
                (2 * rows - 1, 2 * cols - 1), dim=-1
            ).reshape(1, self.token_count, 2)
            centroids = centroids.expand(image.shape[0], -1, -1)
            mass = torch.full(
                (image.shape[0], self.token_count, 1),
                1.0 / self.token_count,
                device=image.device,
                dtype=tokens.dtype,
            )
            metadata = torch.cat((centroids, mass), dim=-1)
        else:
            tokens, metadata = self.pool(features, assignment)
            if self.name == "topology_coords_only":
                tokens = torch.zeros_like(tokens)
        return self.head(
            self.project(tokens),
            metadata,
            adjacency.to(tokens.dtype),
            reachability.to(tokens.dtype),
        )

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


class CrackUNet(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(1, base)
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
        value = F.interpolate(
            bottleneck, size=third.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.decoder3(torch.cat((value, third), dim=1))
        value = F.interpolate(
            value, size=second.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.decoder2(torch.cat((value, second), dim=1))
        value = F.interpolate(
            value, size=first.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.decoder1(torch.cat((value, first), dim=1))
        return self.output(value).squeeze(1)

class MaskTopo(TopoCoarsenModel):
    """Predict connectivity from images and topology built from predicted masks.

    Input: images [B,3,64,64], assignment [B,16,16], and A/R [B,K,K].
    Output: one unnormalized connectivity logit per example.
    """
    def __init__(self, token_count: int = 8, dim: int = 64):
        if not 1 <= token_count <= 256:
            raise ValueError("token_count must be between 1 and 256")
        super().__init__("mask_topo_external", dim, token_count)


MaskPredictor = CrackUNet
GatedTopologicalMessaging = GraphMixLayer
