"""训练 Siamese 网络。"""

import os
import time

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import SiameseMNIST
from model import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss
from test import evaluate


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    # ---------------- data ----------------
    train_set = SiameseMNIST(
        root=cfg.data_root, train=True,
        num_pairs=cfg.num_train_pairs,
        positive_ratio=cfg.positive_ratio,
        seed=cfg.seed,
    )
    val_set = SiameseMNIST(
        root=cfg.data_root, train=False,
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
    embedding_net = EmbeddingNet(embedding_dim=cfg.embedding_dim)
    model = SiameseNet(embedding_net).to(device)

    criterion = ContrastiveLoss(margin=cfg.margin)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (img1, img2, label) in enumerate(train_loader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            e1, e2 = model(img1, img2)          # 前向两次，权重共享
            loss = criterion(e1, e2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % cfg.log_interval == 0:
                print(f"  epoch {epoch} | step {step+1}/{len(train_loader)} "
                      f"| loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # ---------------- validate ----------------
        # evaluate 会先跑一遍 val_loader 收集距离，再扫阈值找最佳准确率
        acc, thr = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] loss {avg_loss:.4f} | val_acc {acc:.4f} "
              f"| best_thr {thr:.3f} | {time.time() - t0:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "embedding_net": embedding_net.state_dict(),
                    "embedding_dim": cfg.embedding_dim,
                    "margin": cfg.margin,
                    "threshold": thr,
                    "val_acc": acc,
                },
                os.path.join(cfg.ckpt_dir, "best.pt"),
            )
            print(f"  -> saved best model (acc={best_acc:.4f})")

    print(f"[done] best val acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()