import torch.nn as nn
import wandb

from src.config import ExperimentConfig
from src.consts import IMG_SIZE, MODELS_DICT
from src.data.preprocess import get_dataloaders, get_datasets
from src.train.train_loop import training_loop
from src.train.eval import generate_preds


class Experiment:
    """
    Class for running the experiment.

    Attributes:
        cfg (ExperimentConfig): The experiment configuration.
    """
    def __init__(self, cfg: ExperimentConfig) -> None:
        """
        Initialize the Experiment.

        Args:
            cfg (ExperimentConfig): The experiment configuration.
        """
        self.cfg = cfg
        if self.cfg.wandb:
            wandb.init(
                project="fashion-mnist",
                name=f"{self.cfg.model}",
                config=vars(self.cfg),
            )

    def run(self) -> None:
        """
        Run the experiment.

        Returns:
            None. Trains the model, evaluates it, and generates predictions.
        """
        train_dataset, test_dataset = get_datasets()
        trainloader, testloader = get_dataloaders(train_dataset, test_dataset, self.cfg.batch_size)
        model = MODELS_DICT[self.cfg.model]
        discriminator = model.d(img_size=IMG_SIZE).to(self.cfg.device)
        generator = model.g(img_size=IMG_SIZE).to(self.cfg.device)
        criterion = nn.BCELoss()
        training_loop(discriminator, generator, trainloader, testloader, criterion, self.cfg, model.unsqueeze)
        generate_preds(generator, train_dataset.classes, self.cfg, model.unsqueeze)
