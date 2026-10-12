# test.py

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from config import cfg
from dataset import get_dataloaders
from model import build_model
from utils import get_device, load_checkpoint


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    acc = correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return acc, all_preds, all_labels


def show_predictions(model, loader, device, classes, n=8):
    model.eval()
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        preds = model(imgs).argmax(dim=1).cpu()

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    plt.figure(figsize=(12, 3))
    for i in range(n):
        img = imgs[i].cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(f"T:{classes[labels[i]]}\nP:{classes[preds[i]]}",
                  fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    device = get_device(cfg.device)
    print("Device:", device)

    _, test_loader = get_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = build_model(num_classes=cfg.num_classes).to(device)

    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重: {ckpt_path}，先跑 train.py")

    load_checkpoint(model, ckpt_path, device=device)

    acc, preds, labels = evaluate(model, test_loader, device)
    print(f"Test Acc: {acc:.4f}")

    classes = test_loader.dataset.classes

    num_classes = len(classes)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    print("Confusion Matrix:")
    print(cm.numpy())

    show_predictions(model, test_loader, device, classes, n=8)


if __name__ == "__main__":
    main()