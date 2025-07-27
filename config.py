from typing import Dict, Any

def get_default_config(mode: str) -> Dict[str, Any]:
    """
    Returns a default configuration dictionary based on the specified training mode.

    Args:
        mode (str): Training mode. Must be one of:
                    - "ae": Autoencoder
                    - "cls": Classification
                    - "seg": Part segmentation

    Returns:
        Dict[str, Any]: A dictionary containing default hyperparameters and settings.

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    base: Dict[str, Any] = {
        "epochs": 100,
        "batch_size": 128,
        "lr": 1e-3,
        "save": True,
        "gpu": 0,
    }

    if mode == "ae":
        return {**base, "model": "PointNetAutoEncoder"}
    elif mode == "cls":
        return {**base, "model": "PointNetCls", "num_classes": 40}
    elif mode == "seg":
        return {**base, "model": "PointNetPartSeg"}
    else:
        raise ValueError(f"Unknown mode: {mode}")
