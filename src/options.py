import argparse

from src.config import ExperimentConfig


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="CGAN")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--hp_tuning", type=bool, default=False)

    args = parser.parse_args()

    cfg = ExperimentConfig(**vars(args))
    return cfg
