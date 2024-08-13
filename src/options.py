import argparse

from src.config import ExperimentConfig


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="CGAN")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--hp_tuning", type=bool, default=True)

    args = parser.parse_args()

    cfg = ExperimentConfig(**vars(args))
    return cfg
