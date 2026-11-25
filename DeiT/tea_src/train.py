
import os

import torch
import torch.nn as nn

from tqdm import tqdm

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    AMP,
    SEED,
    LAST_CHECKPOINT,
    BEST_CHECKPOINT
)

from dataset import build_dataloaders

from model import build_model

from utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint
)


# ============================================================
# 1. Train One Epoch
# ============================================================

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
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Train",
        leave=False
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=AMP and device.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        if scaler.is_enabled():

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# 2. Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(
        loader,
        desc="Val",
        leave=False
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=AMP and device.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# 3. Main
# ============================================================

def main():

    print("=" * 60)
    print("ResNet-18 Teacher Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, _ = build_dataloaders()

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    class_to_idx = train_loader.dataset.class_to_idx

    print(f"Number of classes: {len(class_to_idx)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model().to(DEVICE)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP and DEVICE.type == "cuda"
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_epoch = 0

    best_val_loss = float("inf")

    if os.path.exists(LAST_CHECKPOINT):

        print("\nLoading last checkpoint...")

        checkpoint = load_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            DEVICE
        )

        if checkpoint["class_to_idx"] != class_to_idx:
            raise ValueError(
                "Checkpoint class mapping does not match dataset!"
            )

        start_epoch = checkpoint["epoch"] + 1

        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resume from epoch: {start_epoch + 1}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(start_epoch, EPOCHS):

        print()
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 60)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            DEVICE
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:   {val_acc:.4f}"
        )

        print(
            f"LR: {optimizer.param_groups[0]['lr']:.6e}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        is_best = val_loss < best_val_loss

        if is_best:

            best_val_loss = val_loss

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                class_to_idx
            )

            print("Best checkpoint updated.")

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
            class_to_idx
        )

    print()
    print("=" * 60)
    print("Training Finished")
    print("=" * 60)

    print(f"Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
