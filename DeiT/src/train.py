import os

import torch
import torch.nn as nn

from tqdm import tqdm

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DISTILL_ALPHA,
    AMP,
    SEED,
    TEACHER_CHECKPOINT,
    BEST_CHECKPOINT,
    LAST_CHECKPOINT
)

from dataset import build_dataloaders

from model import (
    build_model,
    build_teacher
)

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
    teacher,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    teacher.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_dist_loss = 0.0

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

        # ----------------------------------------------------
        # Teacher Forward
        # ----------------------------------------------------

        with torch.no_grad():

            with torch.amp.autocast(
                device_type=device.type,
                enabled=AMP and device.type == "cuda"
            ):

                teacher_logits = teacher(images)

                # Hard Distillation:
                # Teacher logits -> predicted class
                teacher_labels = teacher_logits.argmax(
                    dim=1
                )

        # ----------------------------------------------------
        # Student Forward
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type=device.type,
            enabled=AMP and device.type == "cuda"
        ):

            cls_logits, dist_logits = model(images)

            # ------------------------------------------------
            # CLS Head
            # Ground Truth supervision
            # ------------------------------------------------

            cls_loss = criterion(
                cls_logits,
                labels
            )

            # ------------------------------------------------
            # Distillation Head
            # Teacher supervision
            # ------------------------------------------------

            dist_loss = criterion(
                dist_logits,
                teacher_labels
            )

            # ------------------------------------------------
            # Combined Loss
            # ------------------------------------------------

            loss = (
                (1.0 - DISTILL_ALPHA) * cls_loss
                +
                DISTILL_ALPHA * dist_loss
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        if scaler.is_enabled():

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = labels.size(0)

        # DeiT inference:
        # average CLS and Distillation predictions
        ensemble_logits = (
            cls_logits + dist_logits
        ) / 2.0

        total_loss += (
            loss.item() * batch_size
        )

        total_cls_loss += (
            cls_loss.item() * batch_size
        )

        total_dist_loss += (
            dist_loss.item() * batch_size
        )

        total_correct += (
            ensemble_logits.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            dist=f"{dist_loss.item():.4f}"
        )

    return {

        "loss": total_loss / total_samples,

        "cls_loss": (
            total_cls_loss / total_samples
        ),

        "dist_loss": (
            total_dist_loss / total_samples
        ),

        "accuracy": (
            total_correct / total_samples
        )
    }


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

    total_cls_loss = 0.0
    total_dist_loss = 0.0

    total_correct = 0

    total_samples = 0

    progress = tqdm(
        loader,
        desc="Val",
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

        with torch.amp.autocast(
            device_type=device.type,
            enabled=AMP and device.type == "cuda"
        ):

            cls_logits, dist_logits = model(images)

            # ------------------------------------------------
            # Evaluate each Head against Ground Truth
            # ------------------------------------------------

            cls_loss = criterion(
                cls_logits,
                labels
            )

            dist_loss = criterion(
                dist_logits,
                labels
            )

            # ------------------------------------------------
            # Final DeiT Prediction
            # ------------------------------------------------

            ensemble_logits = (
                cls_logits + dist_logits
            ) / 2.0

            loss = criterion(
                ensemble_logits,
                labels
            )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_cls_loss += (
            cls_loss.item() * batch_size
        )

        total_dist_loss += (
            dist_loss.item() * batch_size
        )

        total_correct += (
            ensemble_logits.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    return {

        "loss": total_loss / total_samples,

        "cls_loss": (
            total_cls_loss / total_samples
        ),

        "dist_loss": (
            total_dist_loss / total_samples
        ),

        "accuracy": (
            total_correct / total_samples
        )
    }


# ============================================================
# 3. Main
# ============================================================

def main():

    print("=" * 60)
    print("DeiT Training")
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

    (
        train_loader,
        val_loader,
        _
    ) = build_dataloaders()

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Val samples: "
        f"{len(val_loader.dataset)}"
    )

    class_to_idx = (
        train_loader.dataset.class_to_idx
    )

    # --------------------------------------------------------
    # Student
    # --------------------------------------------------------

    model = build_model().to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Student parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Teacher
    # --------------------------------------------------------

    teacher, teacher_class_to_idx = build_teacher(
        TEACHER_CHECKPOINT,
        DEVICE
    )

    if teacher_class_to_idx != class_to_idx:

        raise ValueError(
            "Teacher and DeiT class mappings "
            "do not match!"
        )

    print()
    print("Teacher loaded successfully.")
    print("Teacher parameters frozen.")

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

    best_val_accuracy = 0.0

    best_val_loss = float("inf")

    if os.path.exists(LAST_CHECKPOINT):

        print()
        print("Loading last checkpoint...")

        checkpoint = load_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            DEVICE
        )

        if (
            checkpoint["class_to_idx"]
            != class_to_idx
        ):

            raise ValueError(
                "Checkpoint class mapping "
                "does not match!"
            )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        # ----------------------------------------------------
        # Backward compatibility:
        # old checkpoint may not contain
        # best_val_accuracy
        # ----------------------------------------------------

        best_val_accuracy = checkpoint.get(
            "best_val_accuracy",
            0.0
        )

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        print(
            f"Resume from epoch: "
            f"{start_epoch + 1}"
        )

        print(
            f"Best Val Accuracy: "
            f"{best_val_accuracy:.4f}"
        )

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        print()
        print("=" * 60)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model,
            teacher,
            train_loader,
            criterion,
            optimizer,
            scaler,
            DEVICE
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Print Metrics
        # ----------------------------------------------------

        print(
            f"Train Loss: "
            f"{train_metrics['loss']:.4f} | "
            f"CLS: "
            f"{train_metrics['cls_loss']:.4f} | "
            f"DIST: "
            f"{train_metrics['dist_loss']:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "
            f"CLS: "
            f"{val_metrics['cls_loss']:.4f} | "
            f"DIST: "
            f"{val_metrics['dist_loss']:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"LR: "
            f"{optimizer.param_groups[0]['lr']:.6e}"
        )

        # ----------------------------------------------------
        # Best Checkpoint
        #
        # New criterion:
        # Validation Accuracy
        # ----------------------------------------------------

        if (
            val_metrics["accuracy"]
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_metrics["accuracy"]
            )

            best_val_loss = (
                val_metrics["loss"]
            )

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

            # Add Best Val Accuracy to checkpoint
            checkpoint = torch.load(
                BEST_CHECKPOINT,
                map_location="cpu",
                weights_only=False
            )

            checkpoint[
                "best_val_accuracy"
            ] = best_val_accuracy

            torch.save(
                checkpoint,
                BEST_CHECKPOINT
            )

            print(
                "Best checkpoint updated."
            )

        # ----------------------------------------------------
        # Last Checkpoint
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

        # Add best validation accuracy
        checkpoint = torch.load(
            LAST_CHECKPOINT,
            map_location="cpu",
            weights_only=False
        )

        checkpoint[
            "best_val_accuracy"
        ] = best_val_accuracy

        torch.save(
            checkpoint,
            LAST_CHECKPOINT
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Training Finished")
    print("=" * 60)

    print(
        f"Best Val Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best Val Loss: "
        f"{best_val_loss:.4f}"
    )


if __name__ == "__main__":

    main()
