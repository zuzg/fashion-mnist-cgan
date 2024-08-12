from pathlib import Path

from src.config import ModelConfig
from src.models.cgan import CDiscriminator, CGenerator
from src.models.dcgan import DCDiscriminator, DCGenerator


# paths
DATA_PATH: Path = Path("data")

# data
IMG_SIZE: int = 28

# models
MODELS_DICT: dict[str, ModelConfig] = {
    "CGAN": ModelConfig(CGenerator, CDiscriminator, True),
    "DCGAN": ModelConfig(DCGenerator, DCDiscriminator, False),
}
