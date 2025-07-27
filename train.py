import argparse
from typing import Optional
from config import get_default_config
from trainers import get_trainer


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for unified PointNet training.

    Returns:
        argparse.Namespace: Parsed arguments containing training mode and optional overrides.
    """
    parser = argparse.ArgumentParser(description="Unified PointNet Training")

    parser.add_argument("--mode", type=str, choices=["ae", "cls", "seg"], required=True,
                        help="Training mode: 'ae' for autoencoder, 'cls' for classification, 'seg' for segmentation.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (-1 for CPU).")
    parser.add_argument("--save", action="store_true", help="Enable checkpoint saving.")

    return parser.parse_args()


def main() -> None:
    """
    Main function to configure, initialize, and launch the training routine.
    """
    args = parse_args()
    config = get_default_config(args.mode)

    # Override default configuration with command-line arguments if provided
    for key in ["epochs", "batch_size", "lr"]:
        val: Optional[int | float] = getattr(args, key)
        if val is not None:
            config[key] = val

    config["gpu"] = args.gpu
    config["save"] = args.save

    trainer_cls = get_trainer(args.mode)
    trainer = trainer_cls(config)
    trainer.train()


if __name__ == "__main__":
    main()
