import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directories():
    for path in (
        config.DATA_ROOT,
        config.DATASET_ROOT,
        config.CHECKPOINT_DIR,
        config.OUTPUT_DIR,
        config.LOG_DIR,
        config.PREDICTION_DIR,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def build_loss():
    # Model returns logits, so use BCEWithLogitsLoss (do not apply sigmoid first).
    return nn.BCEWithLogitsLoss()


def calculate_binary_metrics(logits, targets, threshold=config.THRESHOLD, eps=1e-7):
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    targets = targets >= 0.5

    predictions = predictions.bool()
    targets = targets.bool()

    intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
    pred_sum = predictions.sum(dim=(1, 2, 3)).float()
    target_sum = targets.sum(dim=(1, 2, 3)).float()

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    union = (predictions | targets).sum(dim=(1, 2, 3)).float()
    iou = (intersection + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()


def save_checkpoint(model, optimizer, epoch, val_dice, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice,
            "config": {
                "image_height": config.IMAGE_HEIGHT,
                "image_width": config.IMAGE_WIDTH,
                "image_channels": config.IMAGE_CHANNELS,
                "num_classes": config.NUM_CLASSES,
                "base_channels": config.BASE_CHANNELS,
                "channel_multipliers": config.CHANNEL_MULTIPLIERS,
            },
        },
        path,
    )


def load_checkpoint(model, path, device=config.DEVICE):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def save_training_curves(history, output_path):
    """Save loss/Dice curves without requiring plotting during training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_dice"], label="validation Dice")
    axes[1].plot(epochs, history["val_iou"], label="validation IoU")
    axes[1].set_title("Segmentation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
