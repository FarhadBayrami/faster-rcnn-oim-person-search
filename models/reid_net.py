"""
models/reid_net.py
Re-ID embedding head (ResNet-50 backbone → 256-d L2-normalised embedding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ReIDEmbedder(nn.Module):
    """
    ResNet-50 backbone with a projection head producing a
    256-d L2-normalised embedding suitable for cosine matching.
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Drop the original FC classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → [B, 2048, 1, 1]

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)          # [B, 2048]
        emb  = self.projector(feat)                 # [B, D]
        return F.normalize(emb, dim=1)              # unit hypersphere


class ReIDNet(nn.Module):
    """Wraps embedder + OIM loss for training."""

    def __init__(self, cfg, oim_loss: nn.Module):
        super().__init__()
        self.embedder = ReIDEmbedder(
            embedding_dim=cfg["model"]["embedding_dim"],
            pretrained=True,
        )
        self.oim = oim_loss

    def forward(self, crops: torch.Tensor, pids: torch.Tensor | None = None):
        embs = self.embedder(crops)
        if self.training and pids is not None:
            loss = self.oim(embs, pids)
            return embs, loss
        return embs
