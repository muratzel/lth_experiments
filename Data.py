import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset, Subset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from typing import Tuple, Optional


def load_mnist() -> Tuple[Dataset, Dataset, Dataset, Dataset]:

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

    test_inputs = TensorDataset((test_data.data.unsqueeze(1)/255 - 0.1307)/0.3081)
    test_labels = TensorDataset(test_data.test_labels)

    return train_data, test_data, test_inputs, test_labels


def load_cifar10() -> Tuple[Dataset, Dataset, Dataset, Dataset]:

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4734, 0.4734, 0.4734), (0.2516, 0.2516, 0.2516))])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    test_inputs = TensorDataset((torch.permute(torch.from_numpy(test_data.data), (0, 3, 1, 2)) / 255 - 0.4734) / 0.2516)
    test_labels = TensorDataset(torch.tensor(test_data.targets))

    return train_data, test_data, test_inputs, test_labels


def split_dataset(dataset: Dataset, split: Tuple[int, int]) -> Tuple[Subset, Subset]:

    assert sum(split) == 1.0

    num_data = len(dataset)
    num_split_1 = int(num_data * split[0])
    num_split_2 = num_data - num_split_1

    split_1, split_2 = random_split(dataset, [num_split_1, num_split_2])

    return split_1, split_2


class DataSet:

    def __init__(self, name: str = "mnist", batch_size: Optional[int] = 32, shuffle: Optional[bool] = True,
                 train_val_split: Optional[Tuple[int, int]] = (0.8, 0.2)):

        assert sum(train_val_split) == 1.0

        self.train_data, self.train_loader = None, None
        self.val_data, self.val_loader = None, None
        self.test_data, self.test_loader = None, None

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_val_split = train_val_split

        if name == "mnist":
            self.train_data, self.test_data, self.predict_inputs_data, self.predict_labels_data = load_mnist()
        elif name == "cifar10":
            self.train_data, self.test_data, self.predict_inputs_data, self.predict_labels_data = load_cifar10()
        else:
            raise ValueError("name has to be mnist or cifar10")

        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        self.predict_inputs_loader = DataLoader(self.predict_inputs_data, batch_size=self.batch_size, shuffle=False)
        self.predict_labels_loader = DataLoader(self.predict_labels_data, batch_size=self.batch_size, shuffle=False)

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:

        self.train_data, self.val_data = split_dataset(self.train_data, self.train_val_split)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader

    def get_bootstrap_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:

        num_data = len(self.train_data)

        indices = torch.randint(num_data, (num_data, ))
        bootstrap_train_data = Subset(self.train_data, indices)

        train_data, val_data = split_dataset(bootstrap_train_data, self.train_val_split)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader





