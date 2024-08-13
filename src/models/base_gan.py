from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseGenerator(nn.Module, ABC):
    """
    Base class for the Generator model in a Generative Adversarial Network (GAN).

    Attributes:
        img_size (int): The size of the images.
        label_emb (nn.Embedding): Embedding layer for the labels.
    """
    def __init__(self, img_size: int) -> None:
        """
        Initialize the BaseGenerator.

        Args:
            img_size (int): The size of the images.
        """
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(10, 10)

    @abstractmethod
    def forward(self, z: Tensor, labels: Tensor):
        """
        The forward pass of the generator model.

        Args:
            z (Tensor): The noise vector.
            labels (Tensor): The labels.

        Returns:
            NotImplementedError: This method is not implemented in the base class and should be overridden.
        """
        raise NotImplementedError


class BaseDiscriminator(nn.Module, ABC):
    """
    Base class for the Discriminator model in a Generative Adversarial Network (GAN).

    Attributes:
        dropout (float): Dropout rate for dropout regularization.
        img_size (int): The size of the images.
    """
    def __init__(self, dropout: float, img_size: int) -> None:
        """
        Initialize the BaseDiscriminator.

        Args:
            dropout (float): Dropout rate for dropout regularization.
            img_size (int): The size of the images.
        """
        super().__init__()
        self.dropout = dropout
        self.img_size = img_size

    @abstractmethod
    def forward(self, x: Tensor, labels: Tensor):
        """
        The forward pass of the discriminator model.

        Args:
            x (Tensor): The input tensor.
            labels (Tensor): The labels.

        Returns:
            NotImplementedError: This method is not implemented in the base class and should be overridden.
        """
        raise NotImplementedError
