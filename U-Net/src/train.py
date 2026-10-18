import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from dataset import OxfordPetSegmentationDataset
from model import UNet
from utils import (
    build_loss,
    calculate_binary_metrics,
    ensure_directories,
    save_checkpoint,
    save_training_curves,
    set_seed,
)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    dice_sum = 0.0
    iou_sum = 0.0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in loader:
            images = images.to(config.DEVICE, non_blocking=True)
            masks = masks.to(config.DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            dice, iou = calculate_binary_metrics(logits.detach(), masks)
            total_loss += loss.item() * batch_size
            dice_sum += dice * batch_size
            iou_sum += iou * batch_size
            total_items += batch_size

    denominator = max(total_items, 1)
    return (
        total_loss / denominator,
        dice_sum / denominator,
        iou_sum / denominator,
    )


def main():
    ensure_directories()
    set_seed()

    print(f"Device: {config.DEVICE}")
    print(f"Dataset root: {config.DATASET_ROOT}")

    train_dataset = OxfordPetSegmentationDataset(split="train", augment=True)
    val_dataset = OxfordPetSegmentationDataset(split="val", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = UNet().to(config.DEVICE)
    criterion = build_loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_val_dice = -1.0
    best_path = config.CHECKPOINT_DIR / "unet_oxford_pet_best.pth"
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }
    metrics_path = config.LOG_DIR / "training_metrics.csv"

    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_dice", "val_iou"]
        )

        for epoch in range(1, config.EPOCHS + 1):
            train_loss, _, _ = run_epoch(
                model, train_loader, criterion, optimizer=optimizer
            )
            val_loss, val_dice, val_iou = run_epoch(
                model, val_loader, criterion, optimizer=None
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["val_iou"].append(val_iou)

            print(
                f"Epoch [{epoch}/{config.EPOCHS}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_Dice={val_dice:.4f} "
                f"val_IoU={val_iou:.4f}"
            )
            writer.writerow(
                [epoch, train_loss, val_loss, val_dice, val_iou]
            )
            csv_file.flush()

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                save_checkpoint(
                    model, optimizer, epoch, val_dice, best_path
                )
                print(f"Saved best checkpoint: {best_path}")

    save_training_curves(
        history,
        config.OUTPUT_DIR / "training_curves.png",
    )
    print(f"Training complete. Best validation Dice: {best_val_dice:.4f}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
