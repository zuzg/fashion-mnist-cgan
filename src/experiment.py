import lightning as pl
import wandb

from src.config import ExperimentConfig
from src.consts import IMG_SIZE
from src.data.module import MNISTDataModule
from src.models.dcgan import DCGAN
from src.train.eval import generate_preds
from src.train.hp_tuning import run_hp_tuning


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

    def run(self) -> None:
        """
        Run the experiment.

        Returns:
            None. Trains the model, evaluates it, and generates predictions.
        """
        if self.cfg.hp_tuning:
            run_hp_tuning(self.cfg)
        else:
            if self.cfg.wandb:
                wandb.init(
                    project="fashion-mnist",
                    name=f"{self.cfg.model}",
                    config=vars(self.cfg),
                )
        data_module = MNISTDataModule(self.cfg.batch_size)
        model = DCGAN(IMG_SIZE)
        trainer = pl.Trainer(
            accelerator="auto", devices=1, max_epochs=self.cfg.num_epochs
        )
        trainer.fit(model, data_module)

        generate_preds(model.generator, data_module.classes, self.cfg, model.unsqueeze)
