"""utils/visualise.py — helpers for drawing boxes and showing search results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    pids: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    ax=None,
    title: str = "",
) -> plt.Axes:
    """Draw bounding boxes on a PIL image."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    cmap = plt.get_cmap("tab20")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = cmap(i % 20)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        label_parts = []
        if pids is not None:
            label_parts.append(f"ID:{pids[i]}")
        if scores is not None:
            label_parts.append(f"{scores[i]:.2f}")
        if label_parts:
            ax.text(x1, y1 - 4, " ".join(label_parts),
                    color=color, fontsize=7, weight="bold")
    return ax


def show_search_results(
    query_img: Image.Image,
    gallery_imgs: list[Image.Image],
    gallery_scores: list[float],
    gallery_correct: list[bool],
    top_k: int = 5,
    save_path: str | None = None,
):
    """Display query and top-k gallery results side by side."""
    n = min(top_k, len(gallery_imgs)) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))

    # Query
    axes[0].imshow(query_img)
    axes[0].set_title("Query", fontsize=9)
    axes[0].axis("off")

    for i in range(n - 1):
        axes[i + 1].imshow(gallery_imgs[i])
        color = "green" if gallery_correct[i] else "red"
        axes[i + 1].set_title(f"Rank {i+1}\n{gallery_scores[i]:.3f}",
                               color=color, fontsize=9)
        axes[i + 1].axis("off")
        for spine in axes[i + 1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
