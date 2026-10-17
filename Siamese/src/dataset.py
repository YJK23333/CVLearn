import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class SiameseMNIST(Dataset):
    def __init__(
        self,
        root = "../../data",
        train = True,
        transform = None,
        num_pairs = 30000,
        positive_ratio = 0.5,
        seed = None,
    ):
        super().__init__()

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,)),
            ])
        self.transform = transform

        self.base = datasets.MNIST(
            root=root, train=train, download=False, transform=transform
        )

        self.num_pairs = num_pairs
        self.positive_ratio = positive_ratio

        if seed is not None:
            torch.manual_seed(seed)

        self.targets = np.array(self.base.targets)
        self.classes = np.unique(self.targets)
        self.class_to_indices = {
            c: np.where(self.targets == c)[0] for c in self.classes
        }

    def __len__(self):
        return self.num_pairs

    def _sample_positive(self):
        c = self.classes[torch.randint(len(self.classes), (1,)).item()]
        indices = self.class_to_indices[c]
        n = len(indices)

        i = torch.randint(n, (1,)).item()
        j = torch.randint(n - 1, (1,)).item()
        if j >= i:
            j += 1
        return indices[i], indices[j], 1.0

    def _sample_negative(self):
        c1 = torch.randint(len(self.classes), (1,)).item()
        c2 = torch.randint(len(self.classes) - 1, (1,)).item()
        if(c2 >= c1):
            c2 += 1

        i = self.class_to_indices[self.classes[c1]][
            torch.randint(len(self.class_to_indices[self.classes[c1]]), (1,)).item()
        ]
        j = self.class_to_indices[self.classes[c2]][
            torch.randint(len(self.class_to_indices[self.classes[c2]]), (1,)).item()
        ]
        return i, j, 0.0

    def __getitem__(self, index):
        if torch.rand(1).item() < self.positive_ratio:
            i, j, label = self._sample_positive()
        else:
            i, j, label = self._sample_negative()

        img1, _ = self.base[i]
        img2, _ = self.base[j]
        return img1, img2, torch.tensor(label, dtype=torch.float32)