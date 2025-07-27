import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Optional, Tuple, Literal


def orthogonal_loss(feat_trans: Optional[Tensor], reg_weight: float = 1e-3) -> Tensor:
    """
    Computes a regularisation loss to enforce a transformation matrix to be close to a rotation matrix.
    A rotation matrix A should satisfy A * A^T = I.

    Args:
        feat_trans (Tensor): Batch of transformation matrices of shape (B, K, K).
        reg_weight (float): Weight of the regularisation loss.

    Returns:
        Tensor: Scalar tensor representing the regularisation loss.
    """
    if feat_trans is None:
        return torch.tensor(0.0, device='cpu')

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K, device=device).expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))
    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    y_lengths: Optional[torch.Tensor] = None,
    norm: int = 2,
    batch_reduction: Optional[Literal["mean", "sum"]] = "mean",
    point_reduction: Optional[Literal["mean", "sum", "max"]] = "mean",
    single_directional: bool = False,
    squared: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Optimized Chamfer Distance implementation between two point clouds.
    
    Args:
        x: Tensor of shape (B, P1, D)
        y: Tensor of shape (B, P2, D)
        x_lengths: Optional tensor of shape (B,) with actual lengths of x (used to mask padded points)
        y_lengths: Optional tensor of shape (B,) with actual lengths of y
        norm: 1 (L1) or 2 (L2). Only L2 supports squared.
        batch_reduction: "mean", "sum", or None
        point_reduction: "mean", "sum", "max", or None
        single_directional: if True, only compute x->y
        squared: if True and norm==2, avoids sqrt for speed
        
    Returns:
        Tuple of (distance, None), with distance a scalar if reduced.
    """
    assert x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2], "Batch or feature mismatch"
    B, P1, D = x.shape
    _, P2, _ = y.shape

    # Optionally use squared L2
    if norm == 2 and squared:
        x_sq = (x ** 2).sum(dim=2, keepdim=True)  # (B, P1, 1)
        y_sq = (y ** 2).sum(dim=2, keepdim=True).transpose(1, 2)  # (B, 1, P2)
        dist = x_sq + y_sq - 2 * torch.bmm(x, y.transpose(1, 2))  # (B, P1, P2)
        dist = dist.clamp(min=1e-9)  # Avoid negative due to precision
    else:
        dist = torch.cdist(x, y, p=norm)  # (B, P1, P2)

    # Apply masks for variable-length clouds
    if x_lengths is not None:
        x_mask = torch.arange(P1, device=x.device).unsqueeze(0) >= x_lengths.unsqueeze(1)  # (B, P1)
        dist[x_mask.unsqueeze(2).expand(-1, -1, P2)] = float('inf')
    if y_lengths is not None:
        y_mask = torch.arange(P2, device=y.device).unsqueeze(0) >= y_lengths.unsqueeze(1)  # (B, P2)
        dist[y_mask.unsqueeze(1).expand(-1, P1, -1)] = float('inf')

    x_to_y, _ = dist.min(dim=2)  # (B, P1)
    x_term = reduce_points(x_to_y, point_reduction)  # (B,)

    if not single_directional:
        y_to_x, _ = dist.min(dim=1)  # (B, P2)
        y_term = reduce_points(y_to_x, point_reduction)
        loss = x_term + y_term
    else:
        loss = x_term

    if batch_reduction == "mean":
        loss = loss.mean()
    elif batch_reduction == "sum":
        loss = loss.sum()

    return loss, None


def reduce_points(dists: torch.Tensor, mode: Optional[str]) -> torch.Tensor:
    """Point-wise reduction"""
    if mode == "mean":
        return dists.mean(dim=1)
    elif mode == "sum":
        return dists.sum(dim=1)
    elif mode == "max":
        return dists.max(dim=1).values
    else:
        return dists  # (B, P1/P2)



class Accuracy(nn.Module):
    """
    Computes and accumulates accuracy metric over batches.

    Attributes:
        correct (int): Total correct predictions so far.
        total (int): Total number of predictions processed.
        history (list): List of computed accuracies per epoch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.correct: int = 0
        self.total: int = 0
        self.history: list[torch.Tensor] = []

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Process one batch of predictions and accumulate the number of correct predictions.

        Args:
            preds (torch.Tensor): Predicted labels tensor of shape [B].
            targets (torch.Tensor): Ground truth labels tensor of shape [B].

        Returns:
            torch.Tensor: Accuracy for the current batch as a scalar tensor.
        """
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape."
        with torch.no_grad():
            c = torch.sum(preds == targets)
            t = targets.numel()

            self.correct += c.item()
            self.total += t

        return c.float() / t

    def compute_epoch(self) -> torch.Tensor:
        """
        Compute the accuracy over all batches accumulated so far, store it in history,
        and reset the counters.

        Returns:
            torch.Tensor: Accuracy for the epoch as a scalar tensor.
        """
        acc = torch.tensor(self.correct / self.total, dtype=torch.float32)
        self.history.append(acc)
        self.reset()
        return acc

    def reset(self) -> None:
        """
        Reset the accumulated counters for a new epoch.
        """
        self.correct = 0
        self.total = 0


class mIoU(nn.Module):
    """
    Computes mean Intersection over Union (mIoU) for ShapeNet Part Annotation dataset.
    
    The metric is computed per batch and accumulated over epochs.
    It masks logits based on the class label to compute mIoU only over relevant parts.

    Attributes:
        iou_sum (float): Sum of IoU scores accumulated over batches.
        total (int): Total number of samples processed.
        history (list): List of mIoU scores computed per epoch.
        idx2pids (dict): Mapping from class id to part ids for ShapeNet.
    """

    def __init__(self) -> None:
        super().__init__()
        self.iou_sum: float = 0
        self.total: int = 0
        self.history: list[torch.Tensor] = []

        # Mapping class id to part ids as per ShapeNet Part Anno Dataset
        self.idx2pids = {
            0: [0, 1, 2, 3],
            1: [4, 5],
            2: [6, 7],
            3: [8, 9, 10, 11],
            4: [12, 13, 14, 15],
            5: [16, 17, 18],
            6: [19, 20, 21],
            7: [22, 23],
            8: [24, 25, 26, 27],
            9: [28, 29],
            10: [30, 31, 32, 33, 34, 35],
            11: [36, 37],
            12: [38, 39, 40],
            13: [41, 42, 43],
            14: [44, 45, 46],
            15: [47, 48, 49],
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean IoU over a batch and return the masked predictions.

        Args:
            logits (torch.Tensor): Model output logits of shape [B, 50, num_points].
            targets (torch.Tensor): Ground truth labels of shape [B, num_points].
            class_labels (torch.Tensor): Class label per batch sample, shape [B].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - iou_per_batch (torch.Tensor): Mean IoU over the batch (scalar).
                - batch_masked_pred (torch.Tensor): Masked predictions of shape [B, num_points].
        """
        with torch.no_grad():
            B, N = logits.shape[0], logits.shape[-1]
            device = logits.device
            batch_iou = torch.zeros(B, dtype=torch.float, device=device)
            batch_masked_pred = torch.zeros(B, N, dtype=torch.long, device=device)

            for i in range(B):
                cl = int(class_labels[i].item())
                pids = self.idx2pids[cl]

                logit = logits[i]  # shape [50, num_points]
                target = targets[i]  # shape [num_points]

                # Mask logits to only consider parts for this class
                mask = torch.zeros_like(logit)
                mask[pids, :] = 1
                masked_logit = logit.masked_fill(mask == 0, float("-inf"))

                masked_pred = torch.argmax(masked_logit, dim=0)
                batch_masked_pred[i] = masked_pred

                # Compute IoU for each part
                for pid in pids:
                    pred_part = masked_pred == pid
                    gt_part = target == pid

                    union = (gt_part | pred_part).sum().item()
                    intersection = (gt_part & pred_part).sum().item()

                    if union == 0:
                        batch_iou[i] += 1.0  # perfect score if no points belong to part
                    else:
                        batch_iou[i] += intersection / union

                batch_iou[i] /= len(pids)

            self.iou_sum += batch_iou.sum().item()
            self.total += class_labels.numel()

        iou_per_batch = batch_iou.mean()
        return iou_per_batch, batch_masked_pred

    def compute_epoch(self) -> torch.Tensor:
        """
        Compute the mean IoU over all batches processed in the epoch.

        Returns:
            torch.Tensor: The mean IoU for the epoch as a scalar tensor.
        """
        iou = torch.tensor(self.iou_sum / self.total, dtype=torch.float32)
        self.history.append(iou)
        self.reset()
        return iou

    def reset(self) -> None:
        """
        Reset the accumulated IoU sums and counts for a new epoch.
        """
        self.iou_sum = 0
        self.total = 0
