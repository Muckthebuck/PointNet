import os
import os.path as osp
from typing import List, Tuple
import torch
from torch.nn import Module


class CheckpointManager:
    """
    Manages saving and loading of top-k model checkpoints based on a monitored metric.

    Args:
        dirpath (str): Directory path to save checkpoint files.
        metric_name (str): Name of the metric to track for checkpointing.
        mode (str, optional): One of {'min', 'max'}. Determines if the metric should be minimized or maximized.
                              Defaults to 'min'.
        topk (int, optional): Number of best checkpoints to keep. Defaults to 1.
        verbose (bool, optional): If True, prints info about checkpoint saving/loading. Defaults to False.
    """

    def __init__(
        self,
        dirpath: str,
        metric_name: str,
        mode: str = "min",
        topk: int = 1,
        verbose: bool = False,
    ) -> None:
        self.dirpath: str = dirpath
        self.metric_name: str = metric_name
        self.mode: str = mode
        self.topk: int = topk
        self.verbose: bool = verbose

        self._cache: List[Tuple[str, float]] = []

        os.makedirs(self.dirpath, exist_ok=True)

    def update(self, model: Module, epoch: int, metric: float, fname: str) -> None:
        """
        Save a checkpoint if the metric qualifies within the top-k best values.

        Args:
            model (Module): PyTorch model to save.
            epoch (int): Current epoch number.
            metric (float): Metric value to compare for saving.
            fname (str): Base filename prefix for the checkpoint file.
        """
        assert isinstance(epoch, int), "Epoch must be an integer."
        assert isinstance(metric, float), "Metric must be a float."

        filename = osp.join(self.dirpath, f"{fname}_epoch{epoch}_metric{metric:.4f}.ckpt")

        should_save = False
        if len(self._cache) < self.topk:
            should_save = True
        else:
            if self.mode == "min":
                should_save = any(metric < met for _, met in self._cache)
            elif self.mode == "max":
                should_save = any(metric > met for _, met in self._cache)
            else:
                raise ValueError("mode must be 'min' or 'max'")

        if should_save:
            # Save the checkpoint
            torch.save(model.state_dict(), filename)
            if self.verbose:
                print(f"Saving checkpoint to {filename}")

            self._cache.append((filename, metric))
            # Sort cache and keep only top-k entries
            self._cache.sort(key=lambda x: x[1], reverse=(self.mode == "max"))
            outdated = self._cache[self.topk :]
            self._cache = self._cache[: self.topk]

            # Remove outdated checkpoint files
            for fn, _ in outdated:
                if osp.exists(fn):
                    os.remove(fn)

    def load_best_ckpt(self, model: Module, device: torch.device) -> None:
        """
        Load the best checkpoint (according to the metric) into the given model.

        Args:
            model (Module): Model to load checkpoint weights into.
            device (torch.device): Device to map the checkpoint tensors onto.
        """
        if not self._cache:
            if self.verbose:
                print("No checkpoints available to load.")
            return

        best_ckpt_path = self._cache[0][0]
        try:
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
            if self.verbose:
                print(f"Loaded best checkpoint from {best_ckpt_path}")
        except Exception as e:
            print(f"Failed to load checkpoint from {best_ckpt_path}: {e}")
