from typing import Dict, Any, Tuple
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
# if pytorch3d is installed, use its chamfer_distance
try:
    from pytorch3d.loss.chamfer import chamfer_distance
except ImportError:
    from utils.loss import chamfer_distance
from base_trainer import BaseTrainer
from model import PointNetAutoEncoder
from dataloader import get_data_loaders


class AutoEncoderTrainer(BaseTrainer):
    """
    Trainer class for PointNet AutoEncoder on ModelNet40 dataset.

    Args:
        config (Dict[str, Any]): Configuration dictionary with keys like
            'epochs', 'batch_size', 'lr', 'gpu', 'save', etc.
    """

    def build_model(self) -> Module:
        """Builds and returns the PointNetAutoEncoder model."""
        return PointNetAutoEncoder(num_points=2048)

    def build_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Creates train, validation, and test data loaders using your PointCloudDataset.
        """
        _, dataloaders = get_data_loaders(
            dataset_name="modelnet",
            data_dir="./data",
            batch_size=self.config["batch_size"],
            phases=["train", "val", "test"],
            from_folder=False,
        )
        phase_names = ["train", "val", "test"]
        return dict(zip(phase_names, dataloaders))


    def build_optimizer(self) -> Optimizer:
        """Creates the Adam optimizer for the model parameters."""
        return torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Creates a MultiStepLR scheduler for learning rate decay."""
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[30, 80], gamma=0.5
        )

    def step(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the autoencoder and computes the Chamfer loss.

        Args:
            points (torch.Tensor): Input point clouds, shape [B, N, 3].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loss: Scalar tensor loss.
                - preds: Reconstructed point clouds, shape [B, N, 3].
        """
        points = points.to(self.device)
        preds = self.model(points)
        loss, _ = chamfer_distance(points, preds)
        return loss, preds

    def train_one_epoch(self, epoch: int) -> None:
        """
        Performs a single training epoch on the training dataset.

        Args:
            epoch (int): The current epoch index.
        """
        self.model.train()
        train_loader = self.loaders["train"]
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config['epochs']} - Training")
        for points, _ in pbar:
            points = points.to(self.device)
            self.optimizer.zero_grad()
            loss, preds = self.step(points)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    def validate(self, epoch: int) -> float:
        """
        Runs validation for the current epoch and returns the validation loss.

        Args:
            epoch (int): The current epoch index.

        Returns:
            float: The average validation loss.
        """
        self.model.eval()
        val_loader = self.loaders["val"]
        val_loss = 0.0
        with torch.no_grad():
            for points, _ in val_loader:
                points = points.to(self.device)
                loss, preds = self.step(points)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def test(self) -> None:
        """
        Runs testing on the test dataset and prints final loss.
        """
        self.model.eval()
        test_loader = self.loaders["test"]
        test_loss = 0.0
        with torch.no_grad():
            for points, _ in test_loader:
                points = points.to(self.device)
                loss, preds = self.step(points)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
