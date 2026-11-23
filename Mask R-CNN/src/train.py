
import os
import argparse

import torch

from torch.utils.data import DataLoader

from tqdm import tqdm

import config

from dataset import (
    build_datasets,
    collate_fn
)

from model import (
    build_model,
    count_parameters
)

from utils import (
    get_device,
    move_targets_to_device,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    print_gpu_memory
)


# ============================================================
# 1. Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    amp_enabled
):

    model.train()

    loss_meter = AverageMeter()

    loss_components = {}

    progress = tqdm(
        loader,
        desc=f"Train Epoch {epoch}"
    )

    for images, targets in progress:

        images = [
            image.to(device)
            for image in images
        ]

        targets = move_targets_to_device(
            targets,
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            loss_dict = model(
                images,
                targets
            )

            loss = sum(
                loss_dict.values()
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite training loss: {loss.item()}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP_NORM
        )

        scaler.step(optimizer)

        scaler.update()

        loss_meter.update(
            loss.item()
        )

        for name, value in loss_dict.items():

            if name not in loss_components:

                loss_components[name] = AverageMeter()

            loss_components[name].update(
                value.item()
            )

        progress.set_postfix(
            loss=f"{loss_meter.average:.4f}"
        )

    component_results = {

        name: meter.average

        for name, meter in loss_components.items()
    }

    return loss_meter.average, component_results


# ============================================================
# 2. Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    epoch,
    amp_enabled
):

    # Torchvision detection models return losses
    # only when training mode is enabled.

    model.train()

    loss_meter = AverageMeter()

    loss_components = {}

    progress = tqdm(
        loader,
        desc=f"Val Epoch {epoch}"
    )

    for images, targets in progress:

        images = [
            image.to(device)
            for image in images
        ]

        targets = move_targets_to_device(
            targets,
            device
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            loss_dict = model(
                images,
                targets
            )

            loss = sum(
                loss_dict.values()
            )

        loss_meter.update(
            loss.item()
        )

        for name, value in loss_dict.items():

            if name not in loss_components:

                loss_components[name] = AverageMeter()

            loss_components[name].update(
                value.item()
            )

        progress.set_postfix(
            loss=f"{loss_meter.average:.4f}"
        )

    model.eval()

    component_results = {

        name: meter.average

        for name, meter in loss_components.items()
    }

    return loss_meter.average, component_results


# ============================================================
# 3. Print Loss
# ============================================================

def print_loss(
    title,
    total_loss,
    components
):

    print(f"\n{title}")

    print(
        f"Total Loss: {total_loss:.6f}"
    )

    for name, value in components.items():

        print(
            f"{name}: {value:.6f}"
        )


# ============================================================
# 4. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a new training run"
    )

    args = parser.parse_args()

    config.create_directories()

    config.set_seed()

    device = get_device()

    amp_enabled = (
        config.AMP
        and device.type == "cuda"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        _
    ) = build_datasets()

    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        config.SEED
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        num_workers=config.NUM_WORKERS,

        pin_memory=(
            config.PIN_MEMORY
            and device.type == "cuda"
        ),

        collate_fn=collate_fn,

        generator=loader_generator,

        persistent_workers=False
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=(
            config.PIN_MEMORY
            and device.type == "cuda"
        ),

        collate_fn=collate_fn,

        persistent_workers=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    model.to(device)

    count_parameters(model)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.SGD(

        model.parameters(),

        lr=config.LEARNING_RATE,

        momentum=config.MOMENTUM,

        weight_decay=config.WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=config.SCHEDULER_FACTOR,

        patience=config.SCHEDULER_PATIENCE,

        min_lr=config.MIN_LEARNING_RATE
    )

    # --------------------------------------------------------
    # AMP Scaler
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=amp_enabled
    )

    # --------------------------------------------------------
    # Automatic Resume
    # --------------------------------------------------------

    start_epoch = 1

    best_val_loss = float("inf")

    resume_path = None

    if args.resume:

        resume_path = args.resume

    elif (
        not args.no_resume
        and os.path.exists(config.LAST_CHECKPOINT)
    ):

        resume_path = config.LAST_CHECKPOINT

    if resume_path:

        if not os.path.exists(resume_path):

            raise FileNotFoundError(resume_path)

        start_epoch, best_val_loss = load_checkpoint(

            resume_path,

            model,

            optimizer,

            scheduler,

            scaler,

            device,

            loader_generator
        )

    else:

        print("\nStarting a new training run.")

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    if start_epoch > config.EPOCHS:

        print(
            "Training has already completed."
        )

        return

    print("\nStart Mask R-CNN Training")

    print(
        f"Start epoch: {start_epoch}"
    )

    print(
        f"Total epochs: {config.EPOCHS}"
    )

    print(
        f"AMP enabled: {amp_enabled}"
    )

    for epoch in range(
        start_epoch,
        config.EPOCHS + 1
    ):

        print("\n" + "=" * 70)

        print(
            f"Epoch [{epoch}/{config.EPOCHS}]"
        )

        print("=" * 70)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning Rate: {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss, train_components = train_one_epoch(

            model,

            train_loader,

            optimizer,

            scaler,

            device,

            epoch,

            amp_enabled
        )

        print_loss(
            "Training Results",
            train_loss,
            train_components
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        val_loss, val_components = validate(

            model,

            val_loader,

            device,

            epoch,

            amp_enabled
        )

        print_loss(
            "Validation Results",
            val_loss,
            val_components
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(val_loss)

        # ----------------------------------------------------
        # Best Checkpoint
        # ----------------------------------------------------

        is_best = (
            val_loss < best_val_loss
        )

        if is_best:

            best_val_loss = val_loss

            print(
                "\nNew best validation loss!"
            )

        # ----------------------------------------------------
        # Save Last Checkpoint
        # ----------------------------------------------------

        save_checkpoint(

            config.LAST_CHECKPOINT,

            model,

            optimizer,

            scheduler,

            scaler,

            epoch,

            best_val_loss,

            loader_generator
        )

        print(
            f"Saved: {config.LAST_CHECKPOINT}"
        )

        # ----------------------------------------------------
        # Save Best Checkpoint
        # ----------------------------------------------------

        if is_best:

            save_checkpoint(

                config.BEST_CHECKPOINT,

                model,

                optimizer,

                scheduler,

                scaler,

                epoch,

                best_val_loss,

                loader_generator
            )

            print(
                f"Saved best: {config.BEST_CHECKPOINT}"
            )

        print_gpu_memory()

        print(
            f"\nEpoch {epoch} completed."
        )

        print(
            f"Best Val Loss: {best_val_loss:.6f}"
        )

    print("\n" + "=" * 70)

    print("Training Completed.")

    print(
        f"Best validation loss: {best_val_loss:.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
