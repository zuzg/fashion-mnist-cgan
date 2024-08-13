from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from src.consts import DATA_PATH


def get_datasets() -> tuple[Dataset]:
    """
    Downloads and processes the FashionMNIST dataset.

    Returns:
        tuple[Dataset]: The training and test datasets.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = FashionMNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(DATA_PATH, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def get_dataloaders(train_dataset: Dataset, test_dataset: Dataset, batch_size: int) -> tuple[DataLoader]:
    """
    Creates data loaders for the training and test datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The test dataset.
        batch_size (int): The batch size.

    Returns:
        tuple[DataLoader]: The training and test data loaders.
    """
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
