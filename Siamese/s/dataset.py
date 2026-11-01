"""
Siamese 的数据集，支持 MNIST 和 CIFAR-10。

每次 __getitem__ 随机返回一对：
    (img1, img2, label)
label = 1 → 同类；label = 0 → 不同类
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# 不同数据集的默认 transform（均值和方差用各自的标准值）
_DEFAULT_TRANSFORMS = {
    "mnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]),
}


class SiameseDataset(Dataset):
    def __init__(
        self,
        dataset_name="mnist",
        root="./data",
        train=True,
        transform=None,
        num_pairs=30000,
        positive_ratio=0.5,
        seed=None,
    ):
        super().__init__()

        if dataset_name not in ("mnist", "cifar10"):
            raise ValueError(f"不支持的 dataset: {dataset_name}")
        self.dataset_name = dataset_name

        if transform is None:
            transform = _DEFAULT_TRANSFORMS[dataset_name]
        self.transform = transform

        # ★ 按名字选底层数据集
        if dataset_name == "mnist":
            self.base = datasets.MNIST(
                root=root, train=train, download=True, transform=transform
            )
        else:
            self.base = datasets.CIFAR10(
                root=root, train=train, download=True, transform=transform
            )

        self.num_pairs = num_pairs
        self.positive_ratio = positive_ratio

        if seed is not None:
            torch.manual_seed(seed)

        # ★ 兼容不同 torchvision 版本的字段名
        targets = getattr(self.base, "targets", None)
        if targets is None:
            targets = self.base.labels
        self.targets = np.array(targets)
        self.classes = np.unique(self.targets)
        self.class_to_indices = {
            c: np.where(self.targets == c)[0] for c in self.classes
        }

    def __len__(self):
        return self.num_pairs

    # ---------------- 采样逻辑（和旧版完全一致） ----------------
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
        if c2 >= c1:
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