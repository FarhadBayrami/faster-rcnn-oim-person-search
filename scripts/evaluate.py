"""scripts/evaluate.py — Run Person Search evaluation on the PRW test set."""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import PRWDataset, PRWQueryDataset, get_val_transforms, collate_fn
from models import PersonSearchNet
from utils import load_config, evaluate_search, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/joint.yaml")
    p.add_argument("--checkpoint", default="results/joint/best_model.pth")
    p.add_argument("--det_thresh", type=float, default=0.5,
                   help="Detection score threshold for gallery detections")
    p.add_argument("--top_k", nargs="+", type=int, default=[1, 5, 10])
    return p.parse_args()


@torch.no_grad()
def extract_gallery(model, loader, device, det_thresh):
    """Run detector + embedder on all test frames, collect detections."""
    all_feats, all_pids, all_boxes, all_frames = [], [], [], []
    model.eval()

    for images, targets in tqdm(loader, desc="Gallery"):
        images = [img.to(device) for img in images]
        results = model(images)
        for i, res in enumerate(results):
            scores = res["scores"].cpu().numpy()
            keep   = scores >= det_thresh
            if keep.sum() == 0:
                continue
            feats  = res["embeddings"].cpu().numpy()[keep]
            boxes  = res["boxes"].cpu().numpy()[keep]
            frame  = targets[i]["image_id"].item()

            all_feats.append(feats)
            all_boxes.append(boxes)
            all_pids.append(np.full(keep.sum(), -1))   # pids from detection unknown
            all_frames.extend([frame] * keep.sum())

    return (
        np.vstack(all_feats),
        np.concatenate(all_pids),
        np.vstack(all_boxes),
        np.array(all_frames),
    )


@torch.no_grad()
def extract_queries(model, query_loader, device):
    """Embed all query crops."""
    all_feats, all_pids = [], []
    model.eval()
    for crops, pids in tqdm(query_loader, desc="Queries"):
        crops = crops.to(device)
        feats = model.embedder(crops).cpu().numpy()
        all_feats.append(feats)
        all_pids.extend(pids.numpy().tolist())
    return np.vstack(all_feats), np.array(all_pids)


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = get_device()

    model = PersonSearchNet(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Loaders ──────────────────────────────────────────────────────
    test_ds  = PRWDataset(cfg["data"]["root"], split="test",
                          transforms=get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=4, collate_fn=collate_fn)

    query_ds = PRWQueryDataset(cfg["data"]["root"])
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False, num_workers=4)

    # ── Feature extraction ───────────────────────────────────────────
    g_feats, g_pids, g_boxes, g_frames = extract_gallery(
        model, test_loader, device, args.det_thresh
    )
    q_feats, q_pids = extract_queries(model, query_loader, device)

    # Dummy query boxes / frames (needed for self-exclusion)
    q_boxes  = np.zeros((len(q_feats), 4), dtype=np.float32)
    q_frames = np.full(len(q_feats), -1, dtype=np.int64)  # won't match any frame

    # ── Evaluate ─────────────────────────────────────────────────────
    results = evaluate_search(
        query_feats=q_feats,
        query_pids=q_pids,
        gallery_feats=g_feats,
        gallery_pids=g_pids,
        gallery_boxes=g_boxes,
        gallery_frames=g_frames,
        query_boxes=q_boxes,
        query_frames=q_frames,
        top_k=tuple(args.top_k),
    )

    print("\n── Person Search Results ───────────────────────────────")
    print(f"  mAP    : {results['mAP']*100:.2f}%")
    for k in args.top_k:
        print(f"  Top-{k:<2} : {results[f'top{k}']*100:.2f}%")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
