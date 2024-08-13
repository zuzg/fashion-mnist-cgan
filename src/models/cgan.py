import torch
import torch.nn as nn
from torch import Tensor

from src.models.base_gan import BaseGenerator, BaseDiscriminator


class CGenerator(BaseGenerator):
    """
    Generator class for the GAN. Inherits from the BaseGenerator.

    Attributes:
        img_size (int): The size of the images.
        label_emb (nn.Embedding): Embedding layer for the labels.
        model (nn.Sequential): The sequential model layers.
    """
    def __init__(self, img_size: int):
        """
        Initialize the CGenerator.

        Args:
            img_size (int): The size of the images.
        """
        super().__init__(img_size)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """
        The forward pass of the generator model.

        Args:
            z (Tensor): The noise vector.
            labels (Tensor): The labels.

        Returns:
            Tensor: The generated image.
        """
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), self.img_size, self.img_size)


class CDiscriminator(BaseDiscriminator):
    """
    Discriminator class for the GAN. Inherits from the BaseDiscriminator.

    Attributes:
        img_size (int): The size of the images.
        dropout (float): Dropout rate for dropout regularization.
        label_emb (nn.Embedding): Embedding layer for the labels.
        model (nn.Sequential): The sequential model layers.
    """
    def __init__(self, img_size: int, dropout: float = 0.25):
        """
        Initialize the CDiscriminator.

        Args:
            img_size (int): The size of the images.
            dropout (float): Dropout rate for dropout regularization.
        """
        super().__init__(dropout, img_size)
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """
        The forward pass of the discriminator model.

        Args:
            x (Tensor): The input tensor.
            labels (Tensor): The labels.

        Returns:
            Tensor: The output of the model.
        """
        x = x.view(x.size(0), self.img_size**2)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
