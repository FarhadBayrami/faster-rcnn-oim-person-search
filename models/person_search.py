"""
models/person_search.py
End-to-end Person Search model: detector + re-id head.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.ops import roi_align

from .detector import build_detector
from .reid_net import ReIDEmbedder
from .oim_loss import OIMLoss


class PersonSearchNet(nn.Module):
    """
    Joint detection + re-identification network.

    During training   → returns dict of losses.
    During inference  → returns (detections, embeddings) per image.
    """

    def __init__(self, cfg):
        super().__init__()
        mc = cfg["model"]

        self.detector = build_detector(num_classes=2, pretrained=True)
        self.embedder  = ReIDEmbedder(embedding_dim=mc["embedding_dim"])
        self.oim       = OIMLoss(
            num_features=mc["oim_num_features"],
            num_pids=482,               # PRW train IDs
            scalar=mc["oim_scalar"],
            momentum=mc["oim_momentum"],
        )

        # Loss weights
        lc = cfg.get("loss", {})
        self.w_det  = lc.get("detection_weight", 1.0)
        self.w_reid = lc.get("reid_weight", 0.5)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward(self, images, targets=None):
        if self.training:
            return self._forward_train(images, targets)
        return self._forward_infer(images)

    def _forward_train(self, images, targets):
        # 1. Detection losses (RPN + RoI head)
        det_losses = self.detector(images, targets)

        # 2. Re-ID loss on GT crops
        crops, pids = self._extract_gt_crops(images, targets)
        if crops is not None:
            embs = self.embedder(crops)
            reid_loss = self.oim(embs, pids)
        else:
            reid_loss = torch.tensor(0.0, device=images[0].device)

        total = (
            self.w_det  * sum(det_losses.values())
            + self.w_reid * reid_loss
        )
        return {"loss_total": total, "loss_reid": reid_loss, **det_losses}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_infer(self, images):
        detections = self.detector(images)     # list of dicts
        results = []
        for img, det in zip(images, detections):
            boxes = det["boxes"]
            if boxes.numel() == 0:
                results.append({"boxes": boxes, "scores": det["scores"], "embeddings": torch.empty(0)})
                continue
            crops = self._crop_and_resize(img.unsqueeze(0), boxes)
            embs  = self.embedder(crops)
            results.append({
                "boxes":      boxes,
                "scores":     det["scores"],
                "embeddings": embs,
            })
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_gt_crops(self, images, targets):
        """Crop GT boxes from images for re-id training."""
        all_crops, all_pids = [], []
        for img, tgt in zip(images, targets):
            boxes = tgt["boxes"]
            pids  = tgt["pids"]
            if boxes.numel() == 0:
                continue
            crops = self._crop_and_resize(img.unsqueeze(0), boxes)
            all_crops.append(crops)
            all_pids.append(pids)
        if not all_crops:
            return None, None
        return torch.cat(all_crops), torch.cat(all_pids)

    @staticmethod
    def _crop_and_resize(
        img: torch.Tensor,
        boxes: torch.Tensor,
        output_size: tuple = (256, 128),
    ) -> torch.Tensor:
        """RoI-align crop for each box.  img: [1, C, H, W]."""
        n = boxes.shape[0]
        # roi_align expects [batch_idx, x1, y1, x2, y2]
        batch_idx = torch.zeros(n, 1, device=boxes.device, dtype=boxes.dtype)
        rois = torch.cat([batch_idx, boxes], dim=1)
        return roi_align(img.expand(1, -1, -1, -1), rois, output_size)
