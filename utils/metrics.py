"""
utils/metrics.py
Person Search evaluation: mAP and CMC (top-k recall).
"""

from __future__ import annotations
import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Area under the precision-recall curve (VOC 11-point interpolation)."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p_at_r = precisions[recalls >= thr]
        ap += (p_at_r.max() if p_at_r.size > 0 else 0.0) / 11
    return ap


def evaluate_search(
    query_feats: np.ndarray,      # [Nq, D]
    query_pids:  np.ndarray,      # [Nq]
    gallery_feats: np.ndarray,    # [Ng, D]
    gallery_pids:  np.ndarray,    # [Ng]
    gallery_boxes: np.ndarray,    # [Ng, 4]
    gallery_frames: np.ndarray,   # [Ng]  frame name per detection
    query_boxes: np.ndarray,      # [Nq, 4]
    query_frames: np.ndarray,     # [Nq]  frame name of query crop
    iou_thresh: float = 0.5,
    top_k: tuple = (1, 5, 10),
) -> dict:
    """
    Evaluate Person Search metrics.

    Returns:
        dict with keys 'mAP', 'top1', 'top5', 'top10'
    """
    assert query_feats.shape[1] == gallery_feats.shape[1], "Feature dim mismatch"

    # Cosine similarity (feats should already be L2-normalised)
    sim_matrix = query_feats @ gallery_feats.T   # [Nq, Ng]

    aps, top_k_hits = [], {k: [] for k in top_k}

    for q_idx in range(len(query_feats)):
        q_pid    = query_pids[q_idx]
        q_box    = query_boxes[q_idx]
        q_frame  = query_frames[q_idx]

        sims = sim_matrix[q_idx]
        sorted_idx = np.argsort(-sims)

        # Remove query gallery entries (same frame & overlapping box)
        valid_mask = np.ones(len(gallery_feats), dtype=bool)
        for i, g_idx in enumerate(np.where(gallery_frames == q_frame)[0]):
            if compute_iou(q_box, gallery_boxes[g_idx]) > iou_thresh:
                valid_mask[g_idx] = False

        sorted_idx = sorted_idx[valid_mask[sorted_idx]]
        sorted_pids = gallery_pids[sorted_idx]

        # Positive matches: same PID with good IoU (we rely on PID here)
        correct = (sorted_pids == q_pid)

        # AP
        npos = correct.sum()
        if npos == 0:
            continue
        cum_correct = np.cumsum(correct)
        recalls    = cum_correct / npos
        precisions = cum_correct / (np.arange(len(correct)) + 1)
        aps.append(compute_ap(recalls, precisions))

        # Top-k hits
        for k in top_k:
            top_k_hits[k].append(int(correct[:k].any()))

    mAP = float(np.mean(aps)) if aps else 0.0
    results = {"mAP": mAP}
    for k in top_k:
        vals = top_k_hits[k]
        results[f"top{k}"] = float(np.mean(vals)) if vals else 0.0
    return results
