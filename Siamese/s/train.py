"""训练 Siamese 网络，支持 MNIST / CIFAR-10。"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import SiameseDataset
from model import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss
from test import evaluate


def main():
    cfg = Config()

    # ★ 命令行可以覆盖数据集
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=cfg.dataset,
                        choices=["mnist", "cifar10"])
    args = parser.parse_args()
    cfg.dataset = args.dataset
    cfg.in_channels = 1 if args.dataset == "mnist" else 3   # ★ 自动校正通道数

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}, dataset={cfg.dataset}, "
          f"in_channels={cfg.in_channels}, version={cfg.version}")

    # ---------------- data ----------------
    train_set = SiameseDataset(
        dataset_name=cfg.dataset, root=cfg.data_root, train=True,
        num_pairs=cfg.num_train_pairs,
        positive_ratio=cfg.positive_ratio,
        seed=cfg.seed,
    )
    val_set = SiameseDataset(
        dataset_name=cfg.dataset, root=cfg.data_root, train=False,
        num_pairs=cfg.num_val_pairs,
        positive_ratio=cfg.positive_ratio,
        seed=cfg.seed + 1,
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # ---------------- model ----------------
    embedding_net = EmbeddingNet(
        embedding_dim=cfg.embedding_dim,
        in_channels=cfg.in_channels,       # ★ 传入通道数
    )
    model = SiameseNet(embedding_net).to(device)

    criterion = ContrastiveLoss(margin=cfg.margin)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ★ 文件名带前缀：{version}_{dataset}_best.pt
    prefix = f"{cfg.version}_{cfg.dataset}"
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{prefix}_best.pt")

    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (img1, img2, label) in enumerate(train_loader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            e1, e2 = model(img1, img2)
            loss = criterion(e1, e2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % cfg.log_interval == 0:
                print(f"  epoch {epoch} | step {step+1}/{len(train_loader)} "
                      f"| loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        acc, thr = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] loss {avg_loss:.4f} | val_acc {acc:.4f} "
              f"| best_thr {thr:.3f} | {time.time() - t0:.1f}s")

        if acc > best_acc:
            best_acc = acc
            # ★ checkpoint 存完整信息（坑 5 的正确姿势）
            torch.save(
                {
                    "embedding_net": embedding_net.state_dict(),
                    "embedding_dim": cfg.embedding_dim,
                    "in_channels": cfg.in_channels,
                    "margin": cfg.margin,
                    "threshold": thr,
                    "val_acc": acc,
                    "version": cfg.version,
                    "dataset": cfg.dataset,
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path} (acc={best_acc:.4f})")

    print(f"[done] best val acc = {best_acc:.4f}")
    print(f"       checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()