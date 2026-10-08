import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transfroms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    return train_tf, test_tf


def get_datasets(data_root = "./data"):
    train_tf, test_tf = get_transfroms()

    train_set = datasets.CIFAR10(
        root = data_root, train = True, transform=train_tf
    )
    test_set = datasets.CIFAR10(
        root = data_root, train = False, transform=test_tf
    )

    return train_set, test_set


def get_dataloader(data_root = "./data",
                   batch_size = 128,
                   num_workers = 2):
    train_set, test_set = get_datasets(data_root)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloader("../data")
    imgs, labels = next(iter(train_loader))
    print("image batch:",imgs.shape)
    print("label batch:",labels.shape)
    print("classes:",train_loader.dataset.classes)