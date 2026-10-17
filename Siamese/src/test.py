"""
评估 / 使用训练好的 Siamese 网络。

包含三个部分：
  1. evaluate()          —— 训练时调用的评估函数
  2. plot_distance_hist  —— 画正/负样本对的距离分布
  3. plot_pairs / tsne   —— 可视化样本对和 embedding
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SiameseMNIST
from model import EmbeddingNet, SiameseNet


# ============================================================
# 1. 评估部分（被 train.py 调用）
# ============================================================
@torch.no_grad()
def compute_distances(model, loader, device):
    """跑一遍 loader，收集所有 embedding 的 L2 距离和真实标签。"""
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
    """
    在 [min, max] 上均匀扫阈值：
        距离 < 阈值 → 判为同类；否则判为不同类
    返回 (最佳准确率, 最佳阈值)
    """
    thresholds = np.linspace(distances.min(), distances.max(), num)

    best_acc, best_thr = 0.0, 0.0
    for t in thresholds:
        pred = (distances < t).astype(np.float32)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = float(acc), float(t)

    return best_acc, best_thr


def evaluate(model, loader, device, margin=1.0):
    """训练过程中使用：返回 (准确率, 最佳阈值)。"""
    distances, labels = compute_distances(model, loader, device)
    return best_threshold(distances, labels)


# ============================================================
# 2. 可视化部分（只有直接运行 test.py 时才用到）
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
    print(f"[saved] {save_path}")


def plot_pairs(model, dataset, device, save_path, n=8):
    """随机抽 n 个样本对，画出来并标注距离。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    indices = torch.randperm(len(dataset))[:n].tolist()

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))

    for col, idx in enumerate(indices):
        img1, img2, label = dataset[idx]

        with torch.no_grad():
            e1, e2 = model(img1.unsqueeze(0).to(device),
                           img2.unsqueeze(0).to(device))
            d = F.pairwise_distance(e1, e2).item()

        for row, img in enumerate([img1, img2]):
            ax = axes[row, col]
            ax.imshow(img.squeeze(0).cpu(), cmap="gray")
            ax.axis("off")

        tag = "same" if label.item() > 0.5 else "diff"
        axes[0, col].set_title(f"{tag}\nd={d:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[saved] {save_path}")


def plot_tsne(model, root, device, save_path, n=2000):
    """把 embedding 用 t-SNE 降到 2 维，看看类别是否分得开。"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[skip] 未安装 sklearn，跳过 t-SNE")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root=root, train=False, download=True, transform=tf)

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
    plt.title("t-SNE of Siamese embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[saved] {save_path}")


# ============================================================
# 3. 直接运行 test.py 时的入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./checkpoints/best.pt")
    parser.add_argument("--root", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-pairs", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 加载模型 ----------
    ckpt = torch.load(args.ckpt, map_location=device)
    embedding_net = EmbeddingNet(embedding_dim=ckpt["embedding_dim"]).to(device)
    embedding_net.load_state_dict(ckpt["embedding_net"])
    model = SiameseNet(embedding_net).to(device)
    model.eval()

    print(f"[info] loaded {args.ckpt}, saved val_acc = {ckpt.get('val_acc'):.4f}, "
          f"saved threshold = {ckpt.get('threshold'):.4f}")

    # ---------- 评估 ----------
    val_set = SiameseMNIST(root=args.root, train=False,
                           num_pairs=args.num_pairs, seed=123)
    loader = DataLoader(val_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    distances, labels = compute_distances(model, loader, device)
    acc, thr = best_threshold(distances, labels)
    print(f"[eval] acc = {acc:.4f}, best threshold = {thr:.4f}")

    # ---------- 可视化 ----------
    plot_distance_hist(distances, labels, thr, "./checkpoints/dist_hist.png")
    plot_pairs(model, val_set, device, "./checkpoints/pairs.png", n=8)
    plot_tsne(model, args.root, device, "./checkpoints/tsne.png")


if __name__ == "__main__":
    main()