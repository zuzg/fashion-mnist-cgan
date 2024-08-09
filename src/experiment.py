import torch.nn as nn

from src.config import ExperimentConfig
from src.data.preprocess import get_dataloaders, get_datasets
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.train.train_loop import training_loop
from src.train.eval import generate_preds


class Experiment:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        train_dataset, test_dataset = get_datasets()
        trainloader, testloader = get_dataloaders(train_dataset, test_dataset, self.cfg.batch_size)
        discriminator = Discriminator().to(self.cfg.device)
        generator = Generator().to(self.cfg.device)
        criterion = nn.BCELoss()
        training_loop(discriminator, generator, trainloader, criterion, self.cfg)
        generate_preds(generator, train_dataset.classes, self.cfg.device)
