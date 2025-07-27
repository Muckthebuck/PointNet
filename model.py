import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Dict, Tuple, Optional

class SpatialTransformer(nn.Module):
    """
    Spatial Transformer Network (STN), T-Net module for PointNet.
    """

    def __init__(self, k: int):
        """
        Initialises the Spatial Transformer Network.
        Args:
            k (int): Number of input features.
        """
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the STN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, k, num_points).
        Returns:
            torch.Tensor: Transformed input tensor of shape (batch_size, k, num_points).
        """
        batch_size = x.size(0)
        # global feature extraction
        x = torch.max(self.conv(x), 2)[0]  # Global feature
        # Fully connected layers
        x = self.fc(x) # Expected shape: (batch_size, k*k)

        # add identity matrix bias
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        return x.view(-1, self.k, self.k)

class PointNetFeature(nn.Module):
    """
    PointNet feature extraction module.
    """

    def __init__(self, input_transform=False, feature_transform=False):
        """
        Initialises the PointNet feature extraction module.
        Args:
            input_transform (bool): Whether to use input transformation.
            feature_transform (bool): Whether to use feature transformation.
        """
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        self.stn = SpatialTransformer(3) if self.input_transform else None
        self.fstn = SpatialTransformer(64) if self.feature_transform else None
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        
    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through the PointNet feature extraction module.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, 3).
        Returns:
            Dict[str,Optional[torch.Tensor]]: Dictionary containing 
                                               point feature,
                                               global feature, 
                                               input transformation matrix, and 
                                               feature transformation matrix.
            - 'point_feature' (torch.Tensor): Local point features of shape (batch_size, 64, num_points).
            - 'global_feature' (torch.Tensor): Global feature of shape (batch_size, 1024).
            - 'input_transform' (torch.Tensor, optional): Input transformation matrix of shape (batch_size, 3, 3).
            - 'feature_transform' (torch.Tensor, optional): Feature transformation matrix of shape (batch_size, 64, 64).

        """
        x = x.transpose(2, 1)  # Change shape to (batch_size, 3, num_points)
        if self.input_transform:
            T = self.stn(x)   # Get input transformation matrix
            x = torch.bmm(T, x)  # Apply transformation
        else:
            T = None
        
        point_feat = self.mlp1(x)  # Local feature extraction
        if self.feature_transform:
            F = self.fstn(point_feat)  # Get feature transformation matrix
            point_feat = torch.bmm(F, point_feat)  # Apply feature transformation
        else:
            F = None
        
        x = self.mlp2(point_feat)  # Global feature extraction
        global_feat = torch.max(x, 2)[0]  # Global feature pooling
        return {
            "point_feature": point_feat,
            "global_feature": global_feat,
            "input_transform": T,
            "feature_transform": F
        }
    
class PointNetClassification(nn.Module):
    """
    PointNet classification module.
    """

    def __init__(self, num_classes: int, input_transform=False, feature_transform=False):
        """
        Initialises the PointNet classification module.
        Args:
            num_classes (int): Number of classes for classification.
            input_transform (bool): Whether to use input transformation.
            feature_transform (bool): Whether to use feature transformation.
        """
        super().__init__()
        self.feature_extractor = PointNetFeature(input_transform, feature_transform)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the PointNet classification module.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, 3).
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - logits (torch.Tensor): Output logits of shape (batch_size, num_classes).
            - input_transform (Optional[torch.Tensor]): Input transformation matrix of shape (batch_size, 3, 3) if used.
            - feature_transform (Optional[torch.Tensor]): Feature transformation matrix of shape (batch_size, 64, 64) if used.
        """

        feat_dict = self.feature_extractor(x)
        logits = self.classifier(feat_dict["global_feature"])
        return logits, feat_dict["input_transform"], feat_dict["feature_transform"]
    
class PointNetPartSegmentation(nn.Module):
    """
    PointNet part segmentation module.
    Optionally supports class label conditioning.
    """
    def __init__(self, num_parts: int=50, num_classes=None, embedding_dim: int=16):
        """
        Initialises the PointNet part segmentation module.
        Args:
            num_parts (int): Number of parts for segmentation.
            num_classes (int, optional): Number of classes for class conditioning.
        """
        super().__init__()
        self.feature_extractor = PointNetFeature(input_transform=True, feature_transform=True)
        self.num_parts = num_parts
        self.num_classes = num_classes

        input_channels = 64 + 1024

        if num_classes is not None:
            self.label_embedding = nn.Sequential(
                nn.Linear(num_classes, embedding_dim),
                nn.ReLU()
            )
            input_channels+=embedding_dim
        
        self.segmentation_head = nn.Sequential(
            nn.Conv1d(input_channels, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, num_parts, 1)
        )
    
    def forward(self, x: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the PointNet part segmentation module.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, 3).
            class_labels (Optional[torch.Tensor]): Class labels for conditioning, shape (batch_size, num_classes).
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - logits (torch.Tensor): Output logits of shape (batch_size, num_parts, num_points).
            - input_transform (Optional[torch.Tensor]): Input transformation matrix if used.
            - feature_transform (Optional[torch.Tensor]): Feature transformation matrix if used.
        """
        feat_dict = self.feature_extractor(x)
        global_feature = feat_dict["global_feature"].unsqueeze(2).repeat(1, 1, x.size(1))
        point_feature = feat_dict["point_feature"]
        
        if class_labels is not None:
            class_embedding = self.label_embedding(class_labels)
            class_embedding = class_embedding.unsqueeze(2).repeat(1, 1, x.size(1))
            concat_feat = torch.cat([point_feature, global_feature, class_embedding], dim=1)
        else:
            concat_feat = torch.cat([point_feature, global_feature], dim=1)
        
        logits = self.segmentation_head(concat_feat)
        return logits, feat_dict["input_transform"], feat_dict["feature_transform"]
    

class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points: int = 1024, 
                 latent_dim: int = 1024, 
                 use_skip: bool = True, 
                 use_gaussian_noise: bool = False, 
                 gaussian_noise_std: float = 0.01):
        """
        PointNet-based Autoencoder for point cloud reconstruction.

        Args:
            num_points (int): Number of points in the input/output point clouds.
            latent_dim (int): Size of the global feature vector (latent space).
            use_skip (bool): Whether to include point features as skip connections.
            use_gaussian_noise (bool): Whether to add Gaussian noise to the input for denoising autoencoder training.
            gaussian_noise_std (float): Standard deviation of the Gaussian noise if `use_gaussian_noise` is True.
        """
        super().__init__()
        self.num_points = num_points
        self.use_skip = use_skip
        self.latent_dim = latent_dim
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std

        # Encoder
        self.feature_extractor = PointNetFeature(input_transform=True, feature_transform=True)

        # If using skip connection, input dim = 1024 (global) + 64 (point feature summary)
        decoder_input_dim = latent_dim + 64 if use_skip else latent_dim

        # Decoder (MLP only)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input point cloud of shape (B, N, 3).

        Returns:
            Tuple[reconstructed points, input transform, feature transform]
        """
        # Optionally add Gaussian noise for denoising AE training
        if self.use_gaussian_noise:
            x = x + self.gaussian_noise_std * torch.randn_like(x)

        feat_dict = self.feature_extractor(x)
        global_feat = feat_dict["global_feature"]  # (B, 1024)

        if self.use_skip:
            point_feat = feat_dict["point_feature"]  # (B, 64, N)
            pooled_point_feat = F.adaptive_max_pool1d(point_feat, 1).squeeze(2)  # (B, 64)
            decoder_input = torch.cat([global_feat, pooled_point_feat], dim=1)  # (B, 1088)
        else:
            decoder_input = global_feat  # (B, 1024)

        reconstructed = self.decoder(decoder_input)  # (B, N*3)
        reconstructed = reconstructed.view(-1, self.num_points, 3)  # (B, N, 3)

        return reconstructed, feat_dict["input_transform"], feat_dict["feature_transform"]
