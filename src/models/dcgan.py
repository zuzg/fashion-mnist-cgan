from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from src.models.base_gan import BaseDiscriminator, BaseGenerator


class DCGenerator(BaseGenerator):
    """
    Generator class for the Deep Convolutional GAN (DCGAN). Inherits from the BaseGenerator.

    Attributes:
        img_size (int): The size of the images.
        label_emb (nn.Embedding): Embedding layer for the labels.
        init_size (int): The initial size for the linear layer.
        l1 (nn.Sequential): The first layer of the model.
        model (nn.Sequential): The sequential model layers.
    """

    def __init__(self, img_size: int):
        """
        Initialize the DCGenerator.

        Args:
            img_size (int): The size of the images.
        """
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
        """
        The forward pass of the generator model.

        Args:
            z (Tensor): The noise vector.
            labels (Tensor): The labels.

        Returns:
            Tensor: The generated image.
        """
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.l1(x)
        out = out.view(
            out.size(0), 128, self.init_size, self.init_size
        )  # Reshape to [batch_size, 128, 7, 7]
        img = self.model(out)  # Output shape [batch_size, 1, 28, 28]
        return img


class DCDiscriminator(BaseDiscriminator):
    """
    Discriminator class for the Deep Convolutional GAN (DCGAN). Inherits from the BaseDiscriminator.

    Attributes:
        img_size (int): The size of the images.
        dropout (float): Dropout rate for dropout regularization.
        label_emb (nn.Embedding): Embedding layer for the labels.
        model (nn.Sequential): The sequential model layers.
        adv_layer (nn.Sequential): The final layer of the model.
    """

    def __init__(self, img_size: int, dropout: float = 0.25):
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
        self.adv_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(512 * 2 * 2, 1), nn.Sigmoid()
        )

    def forward(self, img: Tensor, labels: Tensor) -> Tensor:
        """
        The forward pass of the discriminator model.

        Args:
            img (Tensor): The input tensor.
            labels (Tensor): The labels.

        Returns:
            Tensor: The output of the model.
        """
        label_embedding = self.label_emb(labels).view(
            img.size(0), 1, self.img_size, self.img_size
        )
        d_in = torch.cat((img, label_embedding), 1)
        out = self.model(d_in)
        validity = self.adv_layer(out)
        return validity.squeeze()


class DCGAN(pl.LightningModule):
    def __init__(
        self, img_size: int, dropout: float, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = DCGenerator(img_size)
        self.discriminator = DCDiscriminator(img_size, dropout)
        self.example_input_array = (
            torch.zeros(1, img_size, img_size),
            torch.zeros(img_size),
        )

    def forward(self, z: tuple[torch.Tensor, torch.Tensor]) -> Any:
        return self.generator.forward(*z)

    def validation_step(self, *args: Any, **kwargs: Any):
        pass

    def on_validation_epoch_end(self) -> None:
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(
            "validation/generated_images", grid, self.current_epoch
        )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters())
        opt_d = torch.optim.Adam(self.discriminator.parameters())
        return [opt_g, opt_d], []

    def adversarial_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.BCELoss()(y_hat, y)

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        images, labels = batch
        optimizer_g, optimizer_d = self.optimizers()
        z = torch.randn(images.size(0), 100)

        self.toggle_optimizer(optimizer_g)
        self.gen_imgs = self(z)
        sample_imgs = self.gen_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs.unsqueeze(1), nrow=3)
        self.logger.experiment.add_image(
            "train/generated_images", grid, self.current_epoch
        )

        valid = torch.ones(images.size(0), 1)
        g_loss = self.adversarial_loss(self.discriminator(self.gen_imgs), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        self.toggle_optimizer(optimizer_d)

        valid = torch.ones(images.size(0), 1)
        real_loss = self.adversarial_loss(self.discriminator(images), valid)

        fake = torch.ones(images.size(0), 1)
        fake_loss = self.adversarial_loss(
            self.discriminator(self.gen_imgs.detach()), fake
        )
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
