"""
Siamese 网络，支持 MNIST（1 通道 28×28）和 CIFAR-10（3 通道 32×32）。

关键点：SiameseNet 里只有一个 EmbeddingNet 实例，forward 时跑两遍。
"""

import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    """图 → embedding_dim 维向量。"""

    def __init__(self, embedding_dim=128, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),  # ★ in_channels 参数化
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # ★ 不管输入是 28×28 还是 32×32，都压到 7×7
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),   # ★ 这个数对两种数据集都成立
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class SiameseNet(nn.Module):
    def __init__(self, embedding_net: nn.Module):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        e1 = self.embedding_net(x1)
        e2 = self.embedding_net(x2)
        return e1, e2

    def embed(self, x):
        """单张图的 embedding（检索 / 可视化用）。"""
        return self.embedding_net(x)