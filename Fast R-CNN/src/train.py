
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import config

from dataset import (
    VOCDataset,
    collate_fn,
)

from model import FastRCNN

from loss import FastRCNNLoss

from utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
):

    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0

    progress = tqdm(
        dataloader,
        desc="Training",
    )

    for batch in progress:

        image = batch["image"].to(
            config.DEVICE
        )

        proposals = batch["proposals"].to(
            config.DEVICE
        )

        labels = batch["labels"]

        bbox_targets = batch["bbox_targets"]

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=config.DEVICE.type,
            enabled=config.AMP,
        ):

            outputs = model(
                [image],
                [proposals],
            )

            loss, cls_loss, bbox_loss = criterion(
                outputs[0],
                labels,
                bbox_targets,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP_NORM,
        )

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            bbox=f"{bbox_loss.item():.4f}",
        )

    num_batches = max(
        len(dataloader),
        1,
    )

    return (
        total_loss / num_batches,
        total_cls_loss / num_batches,
        total_bbox_loss / num_batches,
    )


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0

    progress = tqdm(
        dataloader,
        desc="Validation",
    )

    for batch in progress:

        image = batch["image"].to(
            config.DEVICE
        )

        proposals = batch["proposals"].to(
            config.DEVICE
        )

        outputs = model(
            [image],
            [proposals],
        )

        loss, cls_loss, bbox_loss = criterion(
            outputs[0],
            batch["labels"],
            batch["bbox_targets"],
        )

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()

    num_batches = max(
        len(dataloader),
        1,
    )

    return (
        total_loss / num_batches,
        total_cls_loss / num_batches,
        total_bbox_loss / num_batches,
    )


def main():

    set_seed()

    print("=" * 60)
    print("Fast R-CNN Training")
    print("=" * 60)

    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"RoIs per image: {config.MAX_ROIS_PER_IMAGE}")

    train_dataset = VOCDataset(
        split="train"
    )

    val_dataset = VOCDataset(
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = FastRCNN().to(
        config.DEVICE
    )

    criterion = FastRCNNLoss()

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_STEP_SIZE,
        gamma=config.LR_GAMMA,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.AMP,
    )

    last_path = (
        config.CHECKPOINT_DIR
        / config.CHECKPOINT_NAME
    )

    best_path = (
        config.CHECKPOINT_DIR
        / config.BEST_CHECKPOINT_NAME
    )

    # ==========================================
    # Resume training
    # ==========================================

    start_epoch = 0
    best_loss = float("inf")

    if last_path.exists():

        print("\nCheckpoint found. Resuming training...")

        checkpoint = load_checkpoint(
            model,
            last_path,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        start_epoch = checkpoint["epoch"] + 1

        best_loss = checkpoint["best_metric"]

        print(
            f"Checkpoint epoch: {checkpoint['epoch'] + 1}"
        )

        print(
            f"Best validation loss: {best_loss:.4f}"
        )

        print(
            f"Resuming from epoch: {start_epoch + 1}"
        )

    else:

        print("\nNo checkpoint found. Starting from scratch.")

    # ==========================================
    # Training
    # ==========================================

    if start_epoch >= config.EPOCHS:

        print("\nTraining has already completed.")
        return

    for epoch in range(start_epoch, config.EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{config.EPOCHS}"
        )

        train_loss, train_cls, train_bbox = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
            )
        )

        val_loss, val_cls, val_bbox = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step()

        print("\nTraining:")
        print(f"Loss: {train_loss:.4f}")
        print(f"Classification: {train_cls:.4f}")
        print(f"Regression: {train_bbox:.4f}")

        print("\nValidation:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Classification: {val_cls:.4f}")
        print(f"Regression: {val_bbox:.4f}")

        # Update best loss before saving last checkpoint.
        is_best = val_loss < best_loss

        if is_best:

            best_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                best_path,
            )

            print(
                f"Best checkpoint saved: {best_path}"
            )

        # Save the latest training state every epoch.
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_loss,
            last_path,
        )

        print(
            f"Latest checkpoint saved: {last_path}"
        )

    print("\nTraining completed.")


if __name__ == "__main__":

    main()
