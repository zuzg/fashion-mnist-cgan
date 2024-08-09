import argparse

from src.config import ExperimentConfig


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", type=bool, default=True)

    args = parser.parse_args()

    cfg = ExperimentConfig(**vars(args))
    return cfg
