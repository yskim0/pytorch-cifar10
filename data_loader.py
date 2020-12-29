import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def train_data_loader(batch_size=256, workers=1, shuffle=True):
    """ return training, test dataloader
    Args:
        batch_size : (int) dataloader batchsize
        workers : (int) # of subprocesses
        shuffle : (bool) data shuffle at every epoch
    Returns:
        train_data_loader : torch dataloader obj.
        test_data_loader : torch dataloader obj.
    """

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                             std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    ])

    train_dataset = datasets.CIFAR10(root='./data/train', train=True, download=True, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers
    )

    return train_data_loader

def test_data_loader(batch_size=256, workers=1, shuffle=True):
    """ return training, test dataloader
    Args:
        batch_size : (int) dataloader batchsize
        workers : (int) # of subprocesses
        shuffle : (bool) data shuffle at every epoch
    Returns:
        train_data_loader : torch dataloader obj.
        test_data_loader : torch dataloader obj.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                             std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    ])

    test_dataset = datasets.CIFAR10(root='./data/test', train=False, download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers
    )

    return test_data_loader