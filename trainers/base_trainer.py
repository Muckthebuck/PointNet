from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from utils.model_checkpoint import CheckpointManager
import datetime

class BaseTrainer(ABC):
    """
    Abstract base class for all PointNet trainers.

    Handles model, optimizer, scheduler, checkpointing, data loaders,
    and the training loop structure.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing
            training parameters such as 'epochs', 'gpu', 'model', 'save', etc.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.device: torch.device = torch.device(
            f"cuda:{config['gpu']}" if config["gpu"] != -1 and torch.cuda.is_available() else "cpu"
        )
        self.model: Module = self.build_model().to(self.device)
        self.optimizer: Optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.checkpoint_manager = self.build_checkpoint_manager() if config.get("save") else None
        self.loaders: Dict[str, DataLoader] = self.build_dataloaders()

    @abstractmethod
    def build_model(self) -> Module:
        """
        Build and return the PyTorch model.

        Returns:
            torch.nn.Module: The model instance.
        """
        pass

    @abstractmethod
    def build_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create and return data loaders for training, validation, and testing.

        Returns:
            Dict[str, DataLoader]: Dictionary with keys like 'train', 'val', 'test' and DataLoader values.
        """
        pass

    @abstractmethod
    def build_optimizer(self) -> Optimizer:
        """
        Build and return the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        pass

    @abstractmethod
    def build_scheduler(self) -> Optional[Any]:
        """
        Build and return the learning rate scheduler.

        Returns:
            Optional[Any]: The scheduler instance or None if not used.
        """
        pass

    @abstractmethod
    def train_one_epoch(self, epoch: int) -> None:
        """
        Perform training logic for a single epoch.

        Args:
            epoch (int): Current epoch number.
        """
        pass

    @abstractmethod
    def validate(self, epoch: int) -> float:
        """
        Perform validation for a single epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Validation metric used for checkpointing (e.g., accuracy or loss).
        """
        pass

    @abstractmethod
    def test(self) -> None:
        """
        Run final evaluation on the test set.
        """
        pass

    def build_checkpoint_manager(self) -> "CheckpointManager":
        """
        Create checkpoint manager instance for saving best model checkpoints.

        Returns:
            CheckpointManager: The checkpoint manager instance.
        """
        path = f"checkpoints/{self.config['model'].lower()}/{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}"
        return CheckpointManager(
            dirpath=path,
            metric_name=self.checkpoint_metric_name(),
            mode=self.checkpoint_mode(),
            topk=2,
            verbose=True,
        )

    def checkpoint_metric_name(self) -> str:
        """
        Metric name used to monitor checkpoints.

        Returns:
            str: Metric name string.
        """
        return "val_loss"

    def checkpoint_mode(self) -> str:
        """
        Mode to optimise the checkpoint metric, either 'min' or 'max'.

        Returns:
            str: 'min' or 'max'.
        """
        return "min"

    def train(self) -> None:
        """
        Run the full training loop including training epochs, validation,
        checkpointing, and final testing.
        """
        for epoch in range(self.config["epochs"]):
            self.train_one_epoch(epoch)
            val_metric = self.validate(epoch)

            if self.checkpoint_manager:
                self.checkpoint_manager.update(self.model, epoch, val_metric)

            if self.scheduler:
                self.scheduler.step()

        if self.checkpoint_manager:
            self.checkpoint_manager.load_best_ckpt(self.model, self.device)

        self.test()
