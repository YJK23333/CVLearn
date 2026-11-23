
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.amp import autocast, GradScaler

from tqdm import tqdm

import config

from dataset import get_dataloaders
from model import FasterRCNN

from loss import compute_losses
from utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch
):

    model.train()

    total_losses = {
        "total": 0.0,
        "rpn_cls": 0.0,
        "rpn_reg": 0.0,
        "roi_cls": 0.0,
        "roi_reg": 0.0,
    }

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}"
    )

    for step, (images, targets) in enumerate(progress):

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
                "image_id": target["image_id"],
            }
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=device.type,
            enabled=config.AMP and device.type == "cuda"
        ):

            outputs = model(
                images,
                targets
            )

            losses = compute_losses(
                outputs,
                targets
            )

            loss = losses["total"]

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP_NORM
        )

        scaler.step(optimizer)
        scaler.update()

        for key in total_losses:

            total_losses[key] += losses[key].item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            rpn_cls=f"{losses['rpn_cls'].item():.4f}",
            roi_cls=f"{losses['roi_cls'].item():.4f}",
        )

    num_batches = len(loader)

    for key in total_losses:

        total_losses[key] /= num_batches

    return total_losses


@torch.no_grad()
def validate(
    model,
    loader,
    device
):

    # Keep training mode so RPN returns training targets.
    # Gradients are disabled during validation.
    model.train()

    total_loss = 0.0

    for images, targets in tqdm(
        loader,
        desc="Validation"
    ):

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
                "image_id": target["image_id"],
            }
            for target in targets
        ]

        with autocast(
            device_type=device.type,
            enabled=config.AMP and device.type == "cuda"
        ):

            outputs = model(
                images,
                targets
            )

            losses = compute_losses(
                outputs,
                targets
            )

        total_loss += losses["total"].item()

    return total_loss / max(len(loader), 1)


def main():

    set_seed()

    device = torch.device(config.DEVICE)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_loader, val_loader, _ = get_dataloaders()

    model = FasterRCNN(
        pretrained=config.PRETRAINED
    ).to(device)

    optimizer = SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = StepLR(
        optimizer,
        step_size=config.LR_STEP_SIZE,
        gamma=config.LR_GAMMA
    )

    scaler = GradScaler(
        "cuda",
        enabled=config.AMP and device.type == "cuda"
    )

    start_epoch, best_loss = load_checkpoint(
        model,
        optimizer,
        scheduler,
        config.LAST_CHECKPOINT
    )

    print("Start epoch:", start_epoch + 1)
    print("Best validation loss:", best_loss)

    for epoch in range(
        start_epoch,
        config.EPOCHS
    ):

        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch
        )

        val_loss = validate(
            model,
            val_loader,
            device
        )

        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        print(
            "Train:",
            {
                key: round(value, 4)
                for key, value in train_losses.items()
            }
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            "Learning Rate:",
            optimizer.param_groups[0]["lr"]
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_loss,
            config.LAST_CHECKPOINT
        )

        if val_loss < best_loss:

            best_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                config.BEST_CHECKPOINT
            )

            print("Best checkpoint updated.")

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":

    main()
