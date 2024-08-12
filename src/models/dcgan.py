import torch
import torch.nn as nn
from torch import Tensor

from src.consts import IMG_SIZE
from src.models.base_gan import BaseGenerator, BaseDiscriminator


class DCGenerator(BaseGenerator):
    def __init__(self, img_size: int = IMG_SIZE):
        super().__init__(img_size)
        self.init_size = self.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(110, 128 * self.init_size**2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.l1(x)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)  # Reshape to [batch_size, 128, 7, 7]
        img = self.model(out)  # Output shape [batch_size, 1, 28, 28]
        return img


class DCDiscriminator(BaseDiscriminator):
    def __init__(self, dropout: float = 0.25, img_size: int = IMG_SIZE):
        super().__init__(dropout, img_size)
        self.label_emb = nn.Embedding(10, 1 * self.img_size**2)

        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
        )

        # Final fully connected layer to output a single scalar for each image
        self.adv_layer = nn.Sequential(nn.Flatten(), nn.Linear(512 * 2 * 2, 1), nn.Sigmoid())

    def forward(self, img: Tensor, labels: Tensor) -> Tensor:
        label_embedding = self.label_emb(labels).view(img.size(0), 1, self.img_size, self.img_size)
        d_in = torch.cat((img, label_embedding), 1)
        out = self.model(d_in)
        validity = self.adv_layer(out)
        return validity.squeeze()
