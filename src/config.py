from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    batch_size: int
    num_epochs: int
    lr: float
    device: str
    wandb: bool
