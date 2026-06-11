"""
models/oim_loss.py
Online Instance Matching (OIM) loss for person re-identification.

Reference:
    Xiao et al., "Joint Detection and Identification Feature Learning for
    Person Search", CVPR 2017.  https://arxiv.org/abs/1604.01850
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OIMLoss(nn.Module):
    """
    OIM loss with a circular lookup table (LUT) for labelled identities
    and a FIFO queue for unlabelled/background instances.

    Args:
        num_features   : embedding dimensionality
        num_pids       : number of labelled person IDs in the training set
        num_unlabelled : size of the circular buffer for unlabelled features
        scalar         : temperature scaling factor (τ)
        momentum       : momentum for updating the LUT
    """

    def __init__(
        self,
        num_features: int = 256,
        num_pids: int = 482,
        num_unlabelled: int = 5000,
        scalar: float = 30.0,
        momentum: float = 0.5,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabelled = num_unlabelled
        self.scalar = scalar
        self.momentum = momentum

        # LUT for labelled identities — NOT parameters (updated with momentum)
        self.register_buffer("lut", torch.zeros(num_pids, num_features))
        # Circular queue for unlabelled instances
        self.register_buffer("queue", torch.zeros(num_unlabelled, num_features))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_lut(self, features: torch.Tensor, pids: torch.Tensor):
        """Momentum update of the lookup table."""
        for feat, pid in zip(features, pids):
            if pid >= 0:
                self.lut[pid] = (
                    self.momentum * self.lut[pid] + (1.0 - self.momentum) * feat
                )
                self.lut[pid] = F.normalize(self.lut[pid], dim=0)

    @torch.no_grad()
    def _enqueue(self, features: torch.Tensor, pids: torch.Tensor):
        """Push unlabelled features into the circular queue."""
        unlabelled = features[pids < 0]
        if unlabelled.numel() == 0:
            return
        ptr = int(self.queue_ptr)
        n = unlabelled.shape[0]
        # wrap-around write
        end = (ptr + n) % self.num_unlabelled
        if ptr + n <= self.num_unlabelled:
            self.queue[ptr : ptr + n] = unlabelled
        else:
            first = self.num_unlabelled - ptr
            self.queue[ptr:] = unlabelled[:first]
            self.queue[: n - first] = unlabelled[first:]
        self.queue_ptr[0] = end

    def forward(self, features: torch.Tensor, pids: torch.Tensor):
        """
        Args:
            features : FloatTensor [N, D] — L2-normalised embeddings
            pids     : LongTensor  [N]    — person IDs (-2 = unlabelled)
        Returns:
            scalar loss
        """
        features = F.normalize(features, dim=1)

        # Concatenate LUT + queue to form the "memory bank"
        memory = torch.cat([self.lut, self.queue], dim=0)   # [K + Q, D]
        logits = self.scalar * features @ memory.t()        # [N, K+Q]

        # Build targets: pid index in [0, num_pids) for labelled, -1 otherwise
        targets = pids.clone()
        targets[pids < 0] = -1  # ignore_index

        loss = F.cross_entropy(logits, targets, ignore_index=-1)

        # Update memory after forward pass
        self._update_lut(features.detach(), pids)
        self._enqueue(features.detach(), pids)

        return loss
