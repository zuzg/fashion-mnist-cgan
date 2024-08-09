import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from src.config import ExperimentConfig


def generator_train_step(
    discriminator: nn.Module,
    generator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    batch_size: int,
    criterion: nn.Module,
    device: str,
) -> float:
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
    criterion: nn.Module,
    cfg: ExperimentConfig,
) -> None:
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.lr)
    d_scheduler = StepLR(d_optimizer, step_size=10, gamma=0.5)
    g_scheduler = StepLR(g_optimizer, step_size=10, gamma=0.5)
    for epoch in range(cfg.num_epochs):
        print(f"Starting epoch {epoch}...")
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
            print(f"g_loss: {g_loss}, d_loss: {d_loss}")
            z = torch.randn(9, 100, device=cfg.device)
            labels = torch.arange(9, device=cfg.device)
            sample_images = generator(z, labels).unsqueeze(1).cpu()
            grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
            plt.imshow(grid)
            plt.show()
