"""
评估 / 使用训练好的 Siamese 网络。

- evaluate：训练时用（train.py 会 import）
- main：直接跑 test.py 时的完整评估 + 可视化
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset import SiameseDataset, _DEFAULT_TRANSFORMS
from model import EmbeddingNet, SiameseNet


# 和 _DEFAULT_TRANSFORMS 里的 Normalize 参数保持一致
_DATASET_STATS = {
    "mnist":   (0.1307, 0.3081),
    "cifar10": ((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
}

# ============================================================
# 1. 评估部分（被 train.py 调用）
# ============================================================
@torch.no_grad()
def compute_distances(model, loader, device):
    model.eval()
    all_dist, all_label = [], []
    for img1, img2, label in loader:
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        e1, e2 = model(img1, img2)
        d = F.pairwise_distance(e1, e2, p=2)
        all_dist.append(d.cpu())
        all_label.append(label)
    return torch.cat(all_dist).numpy(), torch.cat(all_label).numpy()


def best_threshold(distances, labels, num=200):
    thresholds = np.linspace(distances.min(), distances.max(), num)
    best_acc, best_thr = 0.0, 0.0
    for t in thresholds:
        pred = (distances < t).astype(np.float32)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = float(acc), float(t)
    return best_acc, best_thr


def evaluate(model, loader, device, margin=1.0):
    distances, labels = compute_distances(model, loader, device)
    return best_threshold(distances, labels)


# ============================================================
# 2. 可视化
# ============================================================
def plot_distance_hist(distances, labels, threshold, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(distances[labels == 1], bins=60, alpha=0.6,
             density=True, label="positive (same class)")
    plt.hist(distances[labels == 0], bins=60, alpha=0.6,
             density=True, label="negative (diff class)")
    plt.axvline(threshold, color="k", linestyle="--",
                label=f"threshold = {threshold:.3f}")
    plt.xlabel("L2 distance between embeddings")
    plt.ylabel("density")
    plt.title("Distance distribution on validation pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[saved] {save_path}")


def plot_pairs(model, dataset, device, save_path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    indices = torch.randperm(len(dataset))[:n].tolist()
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))

    # ★ 拿当前数据集的 mean / std，用来反归一化
    mean, std = _DATASET_STATS[dataset.dataset_name]
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    for col, idx in enumerate(indices):
        img1, img2, label = dataset[idx]

        with torch.no_grad():
            e1, e2 = model(img1.unsqueeze(0).to(device),
                           img2.unsqueeze(0).to(device))
            d = F.pairwise_distance(e1, e2).item()

        for row, img in enumerate([img1, img2]):
            ax = axes[row, col]
            # ★ 反归一化：x_orig = x_norm * std + mean
            arr = (img.cpu() * std + mean).clamp(0, 1).numpy()
            if arr.shape[0] == 1:
                ax.imshow(arr[0], cmap="gray")
            else:
                ax.imshow(arr.transpose(1, 2, 0))
            ax.axis("off")

        tag = "same" if label.item() > 0.5 else "diff"
        axes[0, col].set_title(f"{tag}\nd={d:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[saved] {save_path}")


def plot_tsne(model, dataset_name, root, device, save_path, n=2000):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[skip] 未安装 sklearn，跳过 t-SNE")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision import datasets

    tf = _DEFAULT_TRANSFORMS[dataset_name]     # ★ 复用 dataset 的默认 transform
    if dataset_name == "mnist":
        ds = datasets.MNIST(root=root, train=False, download=True, transform=tf)
    else:
        ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tf)

    idx = torch.randperm(len(ds))[:n]
    imgs = torch.stack([ds[i][0] for i in idx]).to(device)
    labels = np.array([ds[i][1] for i in idx])

    model.eval()
    with torch.no_grad():
        emb = model.embed(imgs).cpu().numpy()

    xy = TSNE(n_components=2, init="pca", learning_rate=200,
              perplexity=30).fit_transform(emb)

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=labels,
                     cmap="tab10", s=6, alpha=0.8)
    plt.colorbar(sc, ticks=range(10))
    plt.title(f"t-SNE of Siamese embeddings ({dataset_name})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[saved] {save_path}")


# ============================================================
# 3. 直接运行入口
# ============================================================
def main():
    cfg = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="默认 {ckpt_dir}/{version}_{dataset}_best.pt")
    parser.add_argument("--root", default=cfg.data_root)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-pairs", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ★ 默认按前缀拼路径；也可用 --ckpt 指定任意文件
    ckpt_path = args.ckpt or os.path.join(
        cfg.ckpt_dir, f"{cfg.version}_{cfg.dataset}_best.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)

    # ★ 从存档恢复身份信息
    dataset_name = ckpt.get("dataset", cfg.dataset)
    in_channels = ckpt.get("in_channels",
                           1 if dataset_name == "mnist" else 3)
    version = ckpt.get("version", cfg.version)

    embedding_net = EmbeddingNet(
        embedding_dim=ckpt["embedding_dim"], in_channels=in_channels
    ).to(device)
    embedding_net.load_state_dict(ckpt["embedding_net"])
    model = SiameseNet(embedding_net).to(device)
    model.eval()

    saved_acc = ckpt.get("val_acc", float("nan"))
    saved_thr = ckpt.get("threshold", float("nan"))
    print(f"[info] loaded {ckpt_path}")
    print(f"       version={version}, dataset={dataset_name}, "
          f"saved val_acc={saved_acc:.4f}, saved threshold={saved_thr:.4f}")

    # ---------- 评估 ----------
    val_set = SiameseDataset(
        dataset_name=dataset_name, root=args.root, train=False,
        num_pairs=args.num_pairs, seed=123,
    )
    loader = DataLoader(val_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=cfg.num_workers)

    distances, labels = compute_distances(model, loader, device)
    acc, thr = best_threshold(distances, labels)
    print(f"[eval] acc = {acc:.4f}, best threshold = {thr:.4f}")

    # ---------- 可视化（文件名带前缀） ----------
    prefix = f"{version}_{dataset_name}"
    plot_distance_hist(distances, labels, thr,
                       os.path.join(cfg.ckpt_dir, f"{prefix}_dist_hist.png"))
    plot_pairs(model, val_set, device,
               os.path.join(cfg.ckpt_dir, f"{prefix}_pairs.png"), n=8)
    plot_tsne(model, dataset_name, args.root, device,
              os.path.join(cfg.ckpt_dir, f"{prefix}_tsne.png"))


if __name__ == "__main__":
    main()