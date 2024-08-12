from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseGenerator(nn.Module, ABC):
    def __init__(self, img_size: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(10, 10)

    @abstractmethod
    def forward(self, z: Tensor, labels: Tensor):
        raise NotImplementedError


class BaseDiscriminator(nn.Module, ABC):
    def __init__(self, dropout: float, img_size: int) -> None:
        super().__init__()
        self.dropout = dropout
        self.img_size = img_size

    @abstractmethod
    def forward(self, x: Tensor, labels: Tensor):
        raise NotImplementedError
