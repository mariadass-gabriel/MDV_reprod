"""Fashion-MNIST and EMNIST data loaders."""

import ssl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context


def get_fmnist_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 0,
    num_workers: int = 0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))])

    train_full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    val_full   = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_set   = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    # Sequential split matching the paper (first 90% train, last 10% val)
    n_train = int(len(train_full) * (1 - val_fraction))
    train_set = Subset(train_full, list(range(n_train)))
    val_set   = Subset(val_full,   list(range(n_train, len(val_full))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, test_loader


def get_emnist_loader(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,))])
    ood_set = datasets.EMNIST(
        data_dir, split="letters", train=False, download=True, transform=transform)
    return DataLoader(ood_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)
