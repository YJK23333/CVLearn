"""集中管理超参数。"""

import os

# 当前文件在 CVLearn/Siamese/s/config.py
_HERE = os.path.dirname(os.path.abspath(__file__))   # CVLearn/Siamese/s
_SIAMESE_ROOT = os.path.dirname(_HERE)                # CVLearn/Siamese
_CVLEARN_ROOT = os.path.dirname(_SIAMESE_ROOT)        # CVLearn


class Config:
    # ---------- 身份标识（用于文件名前缀） ----------
    version = "s"                  # ★ 改名后改成 "src"

    # ---------- data ----------
    dataset = "mnist"              # ★ "mnist" 或 "cifar10"
    data_root = os.path.join(_CVLEARN_ROOT, "data")   # ★ CVLearn/data
    num_train_pairs = 60000
    num_val_pairs = 10000
    positive_ratio = 0.5

    # ---------- model ----------
    embedding_dim = 256
    in_channels = 1                # ★ mnist=1, cifar10=3

    # ---------- loss ----------
    margin = 1.0

    # ---------- train ----------
    batch_size = 256
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 30
    num_workers = 2
    seed = 42
    log_interval = 100

    # ---------- paths ----------
    ckpt_dir = os.path.join(_SIAMESE_ROOT, "checkpoints")   # ★ CVLearn/Siamese/checkpoints