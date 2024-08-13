from dataclasses import dataclass

from src.models.base_gan import BaseDiscriminator, BaseGenerator


@dataclass
class ExperimentConfig:
    model: str
    batch_size: int
    num_epochs: int
    lr_d: float
    lr_g: float
    device: str
    wandb: bool
    hp_tuning: bool


@dataclass
class ModelConfig:
    g: BaseGenerator
    d: BaseDiscriminator
    unsqueeze: bool
