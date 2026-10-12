# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from dataset import get_dataloaders
from model import build_model
from utils import set_seed, get_device, accuracy, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        c, n = accuracy(logits, labels)
        correct += c
        total += n

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        c, n = accuracy(logits, labels)
        correct += c
        total += n

    return total_loss / total, correct / total


def main():
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print("Device:", device)

    train_loader, test_loader = get_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = build_model(num_classes=cfg.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.model_name))

    best_acc = 0.0
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_best.pth")

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/test", test_acc, epoch)

        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | "
              f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
              f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, ckpt_path)

    writer.close()
    print(f"Best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()