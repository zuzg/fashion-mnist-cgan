from dataclasses import dataclass

from src.models.base_gan import BaseDiscriminator, BaseGenerator


@dataclass
class ExperimentConfig:
    model: str
    batch_size: int
    num_epochs: int
    lr: float
    device: str
    wandb: bool


@dataclass
class ModelConfig:
    g: BaseGenerator
    d: BaseDiscriminator
    unsqueeze: bool
