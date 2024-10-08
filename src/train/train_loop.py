import logging

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from src.config import ExperimentConfig
from src.train.eval import evaluate


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generator_train_step(
    discriminator: nn.Module,
    generator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    batch_size: int,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Performs a training step for the generator.

    Args:
        discriminator (nn.Module): The discriminator model.
        generator (nn.Module): The generator model.
        g_optimizer (torch.optim.Optimizer): The generator's optimizer.
        batch_size (int): The batch size.
        criterion (nn.Module): The loss function.
        device (str): The device type ('cpu' or 'cuda').

    Returns:
        float: The generator loss.
    """
    g_optimizer.zero_grad()
    z = torch.randn(batch_size, 100, device=device)
    fake_labels = torch.randint(0, 10, (batch_size,), device=device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, torch.ones(batch_size, device=device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(
    discriminator: nn.Module,
    generator: nn.Module,
    d_optimizer: torch.optim.Optimizer,
    batch_size: int,
    criterion: nn.Module,
    real_images: Tensor,
    labels: Tensor,
    device: str,
) -> float:
    """
    Performs a training step for the discriminator.

    Args:
        discriminator (nn.Module): The discriminator model.
        generator (nn.Module): The generator model.
        d_optimizer (torch.optim.Optimizer): The discriminator's optimizer.
        batch_size (int): The batch size.
        criterion (nn.Module): The loss function.
        real_images (Tensor): The real images.
        labels (Tensor): The labels for the images.
        device (str): The device type ('cpu' or 'cuda').

    Returns:
        float: The discriminator loss.
    """
    d_optimizer.zero_grad()

    # Train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, torch.ones(batch_size, device=device))

    # Train with fake images
    z = torch.randn(batch_size, 100, device=device)
    fake_labels = torch.randint(0, 10, (batch_size,), device=device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size, device=device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def training_loop(
    discriminator: nn.Module,
    generator: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion: nn.Module,
    cfg: ExperimentConfig,
    unsq: bool,
) -> float:
    """
    The main training loop.

    Args:
        discriminator (nn.Module): The discriminator model.
        generator (nn.Module): The generator model.
        trainloader (DataLoader): The training data loader.
        testloader (DataLoader): The testing data loader.
        criterion (nn.Module): The loss function.
        cfg (ExperimentConfig): The experiment configuration.
        unsq (bool): If True, un-squeeze the sample images.
    Returns:
        float: The loss.
    """
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.lr_g)
    d_scheduler = StepLR(d_optimizer, step_size=10, gamma=0.5)
    g_scheduler = StepLR(g_optimizer, step_size=10, gamma=0.5)
    for epoch in range(cfg.num_epochs):
        log.info(f"Starting epoch {epoch}...")
        generator.train()

        for images, labels in tqdm(trainloader):
            real_images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            batch_size = real_images.size(0)

            d_loss = discriminator_train_step(
                discriminator, generator, d_optimizer, batch_size, criterion, real_images, labels, cfg.device
            )
            g_loss = generator_train_step(discriminator, generator, g_optimizer, batch_size, criterion, cfg.device)

        d_scheduler.step()
        g_scheduler.step()

        generator.eval()
        with torch.no_grad():
            log.info(f"g_loss: {g_loss}, d_loss: {d_loss}")
            z = torch.randn(9, 100, device=cfg.device)
            labels = torch.arange(9, device=cfg.device)
            sample_images = generator(z, labels).cpu()
            if unsq:
                sample_images = sample_images.unsqueeze(1)
            grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
        avg_g_loss, avg_d_loss = evaluate(generator, discriminator, testloader, criterion, cfg.device)
        log.info(f"Test Set - g_loss: {avg_g_loss:.4f}, d_loss: {avg_d_loss:.4f}")
        if cfg.wandb:
            img = wandb.Image(grid)
            wandb.log({"preds": img})
            wandb.log({"metrics/train/G_loss": g_loss, "metrics/train/D_loss": d_loss})
            wandb.log({"metrics/test/G_loss": avg_g_loss, "metrics/test/D_loss": avg_d_loss})
    return g_loss
