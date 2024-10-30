import lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import src.consts
import src.data.preprocess


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 8,
        data_dir: str = str(src.consts.DATA_PATH),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.classes = []
        self.dims = (1, 28, 28)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def prepare_data(self): ...

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = FashionMNIST(
                self.data_dir, train=True, transform=self.transform, download=True
            )
            self.classes = mnist_full.classes

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = FashionMNIST(
                self.data_dir, train=False, transform=self.transform, download=True
            )
            self.classes = self.mnist_test.classes

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)
