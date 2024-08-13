import logging
from functools import partial

import optuna
import torch.nn as nn
from optuna.integration.wandb import WeightsAndBiasesCallback

from src.config import ExperimentConfig
from src.consts import IMG_SIZE, MODELS_DICT
from src.data.preprocess import get_dataloaders, get_datasets
from src.train.train_loop import training_loop


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def objective(trial: optuna.trial.Trial, cfg: ExperimentConfig) -> float:
    lr_d = trial.suggest_float("lr_d", 1e-5, 1e-1, log=True)
    lr_g = trial.suggest_float("lr_g", 1e-5, 1e-1, log=True)
    p = trial.suggest_float("dropout", 0.0, 0.5)

    train_dataset, test_dataset = get_datasets()
    trainloader, testloader = get_dataloaders(train_dataset, test_dataset, cfg.batch_size)
    model = MODELS_DICT[cfg.model]
    discriminator = model.d(img_size=IMG_SIZE, dropout=p).to(cfg.device)
    generator = model.g(img_size=IMG_SIZE).to(cfg.device)
    criterion = nn.BCELoss()
    cfg.lr_d = lr_d
    cfg.lr_g = lr_g
    loss = training_loop(discriminator, generator, trainloader, testloader, criterion, cfg, model.unsqueeze)
    return loss


def run_hp_tuning(cfg: ExperimentConfig) -> None:
    if cfg.wandb:
        wandb_kwargs = {"project": "fashion-mnist", "name": f"{cfg.model}_optuna"}
        wandbc = WeightsAndBiasesCallback(metric_name="loss", wandb_kwargs=wandb_kwargs)
        cb = [wandbc]
    else:
        cb = []
    study = optuna.create_study(direction="minimize")
    obj = partial(objective, cfg=cfg)
    study.optimize(obj, n_trials=2, callbacks=cb)

    log.info("Best trial:")
    trial = study.best_trial
    log.info("  Value: ", trial.value)
    log.info("  Params: ")
    for key, value in trial.params.items():
        log.info(f"    {key}: {value}")
