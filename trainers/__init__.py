from typing import Type
from .ae_trainer import AutoEncoderTrainer
from .cls_trainer import ClassificationTrainer
from .seg_trainer import SegmentationTrainer
from .base_trainer import BaseTrainer 

def get_trainer(mode: str) -> Type[BaseTrainer]:
    """
    Returns the appropriate trainer class based on the training mode.

    Args:
        mode (str): Training mode - one of ["ae", "cls", "seg"].

    Returns:
        Type[BaseTrainer]: A trainer class corresponding to the mode.

    Raises:
        ValueError: If the mode is not recognized.
    """
    if mode == "ae":
        return AutoEncoderTrainer
    elif mode == "cls":
        return ClassificationTrainer
    elif mode == "seg":
        return SegmentationTrainer
    else:
        raise ValueError(f"No trainer available for mode: {mode}")
