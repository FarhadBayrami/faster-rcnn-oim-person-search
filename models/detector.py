"""
models/detector.py
Faster R-CNN pedestrian detector based on torchvision.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Returns a Faster R-CNN with ResNet-50 FPN backbone.

    Args:
        num_classes : 2 = background + pedestrian
        pretrained  : initialise backbone from COCO weights
    """
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        min_size=900,
        max_size=1500,
        box_score_thresh=0.05,
        box_nms_thresh=0.4,
        box_detections_per_img=300,
    )
    # Replace the box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class PedestrianDetector(nn.Module):
    """Thin wrapper exposing a unified interface."""

    def __init__(self, cfg):
        super().__init__()
        self.model = build_detector(
            num_classes=cfg["model"]["num_classes"],
            pretrained=cfg["model"]["pretrained"],
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def load_checkpoint(self, path: str, device="cpu"):
        state = torch.load(path, map_location=device)
        self.model.load_state_dict(state["model"])
        print(f"[Detector] Loaded checkpoint from {path}")
