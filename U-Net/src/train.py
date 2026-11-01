
import csv
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import config
from model import build_model
from dataset import get_dataloaders
from utils import (
    BCEDiceLoss,
    calculate_metrics,
    save_checkpoint,
    load_checkpoint
)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    total_loss = 0.0

    progress = tqdm(loader, desc="Train")

    for images, masks in progress:

        images = images.to(
            device, non_blocking=True
        )

        masks = masks.to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type="cuda",
            enabled=config.USE_AMP and device.type == "cuda"
        ):

            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    total_intersection = 0.0
    total_pred = 0.0
    total_target = 0.0
    total_correct = 0
    total_pixels = 0

    progress = tqdm(loader, desc="Validation")

    for images, masks in progress:

        images = images.to(
            device, non_blocking=True
        )

        masks = masks.to(
            device, non_blocking=True
        )

        with autocast(
            device_type="cuda",
            enabled=config.USE_AMP and device.type == "cuda"
        ):

            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()

        preds = (
            torch.sigmoid(outputs) >= config.THRESHOLD
        ).long()

        targets = masks.long()

        total_intersection += (
            (preds * targets).sum().item()
        )

        total_pred += preds.sum().item()
        total_target += targets.sum().item()

        total_correct += (
            (preds == targets).sum().item()
        )

        total_pixels += targets.numel()

    dice = (
        2 * total_intersection + 1e-6
    ) / (
        total_pred + total_target + 1e-6
    )

    union = (
        total_pred + total_target - total_intersection
    )

    iou = (
        total_intersection + 1e-6
    ) / (
        union + 1e-6
    )

    pixel_accuracy = total_correct / total_pixels

    return {
        "loss": total_loss / len(loader),
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": pixel_accuracy
    }


def main():

    device = config.DEVICE

    print("=" * 60)
    print("U-Net Baseline Training")
    print(f"Device: {device}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Image Size: {config.IMAGE_SIZE}")
    print("=" * 60)

    # 1. Dataset
    train_loader, val_loader, _ = get_dataloaders()

    # 2. Model
    model = build_model().to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Model parameters: {total_params:,}")

    # 3. Loss
    criterion = BCEDiceLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )

    # 4. Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    scaler = GradScaler(
        "cuda",
        enabled=config.USE_AMP and device.type == "cuda"
    )

    # 5. Resume
    start_epoch = 1
    best_dice = 0.0

    if config.RESUME and config.LAST_MODEL_PATH.exists():

        model, optimizer, start_epoch, best_dice = (
            load_checkpoint(
                model,
                optimizer,
                config.LAST_MODEL_PATH,
                device
            )
        )

        print(f"Resume from epoch {start_epoch}")

    # 6. Training log
    log_path = config.RUN_DIR / "training_log.csv"

    write_header = (
        not log_path.exists() or start_epoch == 1
    )

    with open(
        log_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice",
                "val_iou",
                "val_pixel_accuracy",
                "learning_rate"
            ])

        # 7. Epoch loop
        for epoch in range(start_epoch, config.EPOCHS + 1):

            start_time = time.time()

            print(f"\nEpoch [{epoch}/{config.EPOCHS}]")

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device
            )

            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device
            )

            scheduler.step(val_metrics["dice"])

            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            print("-" * 50)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f}")
            print(f"Dice:       {val_metrics['dice']:.4f}")
            print(f"IoU:        {val_metrics['iou']:.4f}")
            print(
                f"Pixel Acc:  "
                f"{val_metrics['pixel_accuracy']:.4f}"
            )
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Time: {elapsed:.1f}s")

            writer.writerow([
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["dice"],
                val_metrics["iou"],
                val_metrics["pixel_accuracy"],
                current_lr
            ])

            f.flush()

            # Save last checkpoint
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_dice,
                config.LAST_MODEL_PATH
            )

            # Save best model
            if val_metrics["dice"] > best_dice:

                best_dice = val_metrics["dice"]

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_dice,
                    config.BEST_MODEL_PATH
                )

                print(
                    f"Best model updated! "
                    f"Dice: {best_dice:.4f}"
                )

    print("\nTraining finished.")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best model: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
