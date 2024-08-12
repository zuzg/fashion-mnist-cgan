from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseGenerator(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)

    @abstractmethod
    def forward(self, z: Tensor, labels: Tensor):
        raise NotImplementedError


class BaseDiscriminator(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor, labels: Tensor):
        raise NotImplementedError
