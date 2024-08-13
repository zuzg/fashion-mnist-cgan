import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.config import ExperimentConfig


def evaluate(
    generator: nn.Module, discriminator: nn.Module, testloader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float]:
    """
    Evaluate the generator and discriminator models using the given test data.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        testloader (DataLoader): The test data loader.
        criterion (nn.Module): The loss function.
        device (str): The device type ('cpu' or 'cuda').

    Returns:
        tuple[float]: The average generator and discriminator losses.
    """
    generator.eval()
    discriminator.eval()

    g_losses = []
    d_losses = []

    with torch.no_grad():
        for images, labels in testloader:
            real_images = images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            # Discriminator loss on real images
            real_validity = discriminator(real_images, labels)
            real_loss = criterion(real_validity, torch.ones(batch_size, device=device))

            # Generate fake images
            z = torch.randn(batch_size, 100, device=device)
            fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            fake_images = generator(z, fake_labels)

            # Discriminator loss on fake images
            fake_validity = discriminator(fake_images, fake_labels)
            fake_loss = criterion(fake_validity, torch.zeros(batch_size, device=device))

            # Total discriminator loss
            d_loss = real_loss + fake_loss
            d_losses.append(d_loss.item())

            # Generator loss
            validity = discriminator(fake_images, fake_labels)
            g_loss = criterion(validity, torch.ones(batch_size, device=device))
            g_losses.append(g_loss.item())

    avg_g_loss = np.mean(g_losses)
    avg_d_loss = np.mean(d_losses)

    return avg_g_loss, avg_d_loss


def generate_preds(generator: nn.Module, classes: list[str], cfg: ExperimentConfig, unsq: bool):
    """
    Generate predictions using the generator model.

    Args:
        generator (nn.Module): The generator model.
        classes (list[str]): The list of classes.
        cfg (ExperimentConfig): The experiment configuration.
        unsq (bool): If True, un-squeeze the sample images.

    Returns:
        None. Saves the prediction grid as an image and logs to Weights and Biases if enabled.
    """
    # Generate latent vectors (z) and corresponding labels
    z = torch.randn(100, 100, device=cfg.device)
    labels = torch.tensor([i for _ in range(10) for i in range(10)], device=cfg.device)

    # Generate sample images using the generator
    generator.eval()
    with torch.no_grad():
        sample_images = generator(z, labels).cpu()
        if unsq:
            sample_images = sample_images.unsqueeze(1)
    grid = make_grid(sample_images, nrow=10, normalize=True).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid)
    plt.yticks([])
    plt.xticks(
        np.arange(15, 300, 30),
        classes,
        rotation=45,
        fontsize=20,
    )
    if cfg.wandb:
        wandb.log({"final_preds": plt})
